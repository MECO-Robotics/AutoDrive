import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from frc_defense.tensor_training import (
    _dispatch_opponent_evaluations,
    _opponent_evaluation_paths,
)


def test_parallel_cpu_evaluation_dispatch_preserves_seed_paths_and_result_order():
    specs = [
        {"name": "scripted", "seed": 501, "output": "/eval/scripted.json"},
        {"name": "adstar", "seed": 501, "output": "/eval/adstar.json"},
        {"name": "peer", "seed": 501, "output": "/eval/peer.json"},
    ]

    def evaluate(index, spec, stream):
        assert stream is None
        # Force completion order to differ from input order.
        time.sleep((len(specs) - index) * .005)
        return {"name": spec["name"], "seed": spec["seed"],
                "output": spec["output"], "episodes": 4}

    serial = _dispatch_opponent_evaluations(specs, evaluate, workers=1, device="cpu")
    parallel = _dispatch_opponent_evaluations(specs, evaluate, workers=3, device="cpu")

    assert parallel == serial
    assert [item["name"] for item in parallel] == [item["name"] for item in specs]
    assert [item["seed"] for item in parallel] == [501, 501, 501]
    assert [item["output"] for item in parallel] == [item["output"] for item in specs]


def test_evaluation_dispatch_rejects_zero_workers():
    with pytest.raises(ValueError, match="evaluation_workers"):
        _dispatch_opponent_evaluations([], lambda *_: None, workers=0)


def test_concurrent_evaluation_uses_private_side_effect_paths_and_same_report_path():
    spec = {"name": "NN · peer / current", "mode": "learned"}
    canonical_serial, execution_serial = _opponent_evaluation_paths(
        Path("/eval/generation-0001"), spec, 2, False)
    canonical_parallel, execution_parallel = _opponent_evaluation_paths(
        Path("/eval/generation-0001"), spec, 2, True)

    assert canonical_parallel == canonical_serial
    assert execution_serial == canonical_serial
    assert execution_parallel != canonical_parallel
    assert execution_parallel == Path(
        "/eval/generation-0001/.workers/worker-002/metrics.json")
