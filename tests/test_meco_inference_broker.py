"""Deterministic event-driven coverage for the production async broker contract."""
import threading
import time
from types import SimpleNamespace

import torch

from frc_defense.meco_tracks import _AsyncInferenceBroker


class FakeAdapter:
    def __init__(self, infer):
        self.track = "C1"
        self._generation_id = "generation-1"
        self._request_sequence = 0
        self.checkpoint_sha256 = "a" * 64
        self.protocol_id = "frozen-protocol"
        self.infer = infer

    def request(self, *, sequence=None, generation=None, expiry=None):
        if sequence is None:
            self._request_sequence += 1
            sequence = self._request_sequence
        else:
            self._request_sequence = max(self._request_sequence, sequence)
        return SimpleNamespace(
            generation_id=generation or self._generation_id,
            request_sequence=sequence,
            checkpoint_sha256=self.checkpoint_sha256,
            protocol_id=self.protocol_id,
            model_id="meco:C1",
            snapshot_id=f"snapshot-{sequence}",
            expires_monotonic=(time.monotonic() + 10 if expiry is None else expiry),
        )

    def _infer_sync(self, request, inputs, *, now):
        return self.infer(request, inputs, now=now)


def response_for(request, outputs=None):
    return SimpleNamespace(request_sequence=request.request_sequence,
        generation_id=request.generation_id, snapshot_id=request.snapshot_id,
        completed_monotonic=time.monotonic(), outputs=outputs or {
            "candidate_utility": torch.tensor([0.0, 1.0])})


def test_newest_pending_request_replaces_older_queued_request():
    started, release = threading.Event(), threading.Event()
    calls = []

    def infer(request, _inputs, *, now):
        calls.append(request.request_sequence)
        if request.request_sequence == 1:
            started.set()
            assert release.wait(2)
        return response_for(request)

    adapter = FakeAdapter(infer)
    broker = _AsyncInferenceBroker(adapter)
    try:
        r1 = adapter.request()
        h1 = broker.submit(r1, {"x": torch.tensor([1.0])})
        assert h1 is not None and started.wait(1)
        r2 = adapter.request()
        h2 = broker.submit(r2, {"x": torch.tensor([2.0])})
        r3 = adapter.request()
        h3 = broker.submit(r3, {"x": torch.tensor([3.0])})
        assert h2 is not None and h3 is not None
        assert h2["discarded"]
        assert broker.counters()["dropped_requests"] == 1
        release.set()
        broker.pending.join()
        assert calls == [1, 3]
        outcomes = broker.counters()["recent_outcomes"]
        assert any(row["request_sequence"] == 2 and row["status"] == "superseded"
                   for row in outcomes)
        assert broker.poll(r1, h1) is None
        assert broker.poll(r3, h3) is not None
        published = next(row for row in broker.counters()["recent_outcomes"]
                         if row["request_sequence"] == 3)
        assert published["status"] == "consumed"
        assert published["events"] == ["queued", "started", "published", "consumed"]
        assert published["durations_s"]["inference"] >= 0
    finally:
        release.set()
        broker.close()


def test_submit_and_poll_reject_mismatched_identity():
    adapter = FakeAdapter(lambda req, _inputs, now: response_for(req))
    broker = _AsyncInferenceBroker(adapter)
    try:
        wrong_generation = adapter.request(generation="other-generation")
        assert broker.submit(wrong_generation, {"x": torch.tensor([0.0])}) is None
        assert broker.counters()["inference_faults"] == 1

        request = adapter.request()
        handle = broker.submit(request, {"x": torch.tensor([1.0])})
        assert handle is not None
        broker.pending.join()
        forged = SimpleNamespace(**vars(request))
        forged.snapshot_id = "another-snapshot"
        assert broker.poll(forged, handle) is None
        assert broker.counters()["recent_outcomes"][-1]["status"] == "identity_mismatch"
        result = broker.poll(request, handle)
        assert result is not None
        assert broker.poll(request, handle) is None
        for _ in range(140):
            stale = adapter.request(generation="stale-generation")
            assert broker.submit(stale, {"x": torch.tensor([0.0])}) is None
        outcomes = broker.counters()["recent_outcomes"]
        assert len(outcomes) == 128
        assert all("outputs" not in row and "tensor" not in row for row in outcomes)
    finally:
        broker.close()


def test_poll_before_completion_consumes_handle_and_prevents_late_reuse():
    started, release = threading.Event(), threading.Event()

    def infer(request, _inputs, *, now):
        started.set()
        assert release.wait(2)
        return response_for(request)

    adapter = FakeAdapter(infer)
    broker = _AsyncInferenceBroker(adapter)
    try:
        request = adapter.request()
        handle = broker.submit(request, {"x": torch.tensor([1.0])})
        assert handle is not None and started.wait(1)
        assert broker.poll(request, handle) is None
        release.set()
        broker.pending.join()
        assert handle["response"] is None
        assert broker.poll(request, handle) is None
        assert broker.counters()["late_responses"] == 1
        outcome = broker.counters()["recent_outcomes"][-1]
        assert outcome["status"] == "consumed_before_completion"
        assert outcome["events"] == ["queued", "started", "consumed"]
    finally:
        release.set()
        broker.close()


def test_expired_none_and_raised_inference_faults_are_isolated():
    calls = []

    def infer(request, _inputs, *, now):
        calls.append(request.request_sequence)
        if request.request_sequence == 2:
            return None
        if request.request_sequence == 3:
            raise RuntimeError("injected scorer fault")
        return response_for(request)

    adapter = FakeAdapter(infer)
    broker = _AsyncInferenceBroker(adapter)
    try:
        expired = adapter.request(expiry=time.monotonic() - 1)
        expired_handle = broker.submit(expired, {"x": torch.tensor([0.0])})
        assert expired_handle is not None
        broker.pending.join()
        assert expired_handle["done"] and expired_handle["response"] is None
        assert calls == []
        assert broker.counters()["late_responses"] == 1
        assert broker.counters()["recent_outcomes"][-1]["status"] == "expiry_before_start"

        missing = adapter.request()
        missing_handle = broker.submit(missing, {"x": torch.tensor([0.0])})
        assert missing_handle is not None
        broker.pending.join()
        assert broker.poll(missing, missing_handle) is None
        assert broker.counters()["inference_faults"] == 1
        assert broker.counters()["recent_outcomes"][-1]["status"] == "model_none"

        raised = adapter.request()
        raised_handle = broker.submit(raised, {"x": torch.tensor([0.0])})
        assert raised_handle is not None
        broker.pending.join()
        assert broker.poll(raised, raised_handle) is None
        assert broker.counters()["inference_faults"] == 2
        assert broker.counters()["recent_outcomes"][-1]["status"] == "model_exception"

        subsequent = adapter.request()
        subsequent_handle = broker.submit(subsequent, {"x": torch.tensor([1.0])})
        assert subsequent_handle is not None
        broker.pending.join()
        assert broker.poll(subsequent, subsequent_handle) is not None
        assert calls == [2, 3, 4]
    finally:
        broker.close()


def test_close_is_idempotent_joins_worker_and_rejects_new_requests():
    started, release, closing = threading.Event(), threading.Event(), threading.Event()

    def infer(request, _inputs, *, now):
        started.set()
        assert release.wait(2)
        return response_for(request)

    adapter = FakeAdapter(infer)
    broker = _AsyncInferenceBroker(adapter)
    request = adapter.request()
    handle = broker.submit(request, {"x": torch.tensor([1.0])})
    assert handle is not None and started.wait(1)
    closer = threading.Thread(target=lambda: (closing.set(), broker.close()))
    closer.start()
    assert closing.wait(1)
    release.set()
    closer.join(timeout=2)
    assert not closer.is_alive()
    assert not broker.thread.is_alive()
    broker.close()
    assert broker.counters()["closed"]
    rejected = adapter.request()
    assert broker.submit(rejected, {"x": torch.tensor([2.0])}) is None
