"""Bounded archival evidence kept outside the synchronous control path."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def load_bound_reference(path: Path, expected_sha256: str):
    """Read and verify an evaluation reference before entering the control loop."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Archived reference SHA256 mismatch')
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError('Handoff reference must be an object')
    return value


def return_attribution(return_actor) -> dict:
    """Attribute RETURN navigation to the actual actor reference, not an arm alias."""
    present = return_actor is not None
    return dict(mode='conditional-return-'+('neural' if present else 'feedback'),
                navigation_input_to_NN=present)


def validate_return_attribution(record: dict, *, actor_present: bool) -> None:
    """Independent strict-type check against observed actor-presence evidence."""
    if type(actor_present) is not bool:
        raise ValueError('Actor presence must be an observed boolean')
    actual = record.get('control_attribution')
    if not isinstance(actual, dict) or set(actual) != {'mode', 'navigation_input_to_NN'}:
        raise ValueError('Missing or malformed RETURN attribution')
    flag = actual['navigation_input_to_NN']
    mode = 'conditional-return-neural' if actor_present else 'conditional-return-feedback'
    if type(flag) is not bool or flag is not actor_present or actual['mode'] != mode:
        raise ValueError('RETURN attribution disagrees with actual actor evidence')


def validate_return_run(records, *, actor_present: bool, start_file_sha256, end_file_sha256,
                        start_state_sha256, end_state_sha256, actual_forwards: int,
                        return_decisions: int) -> None:
    """Reconcile frame attribution with actual start/end hashes and call counts.

    Callers must obtain these hashes from the actual loaded actor and verify
    their expected checkpoint pins separately. This does not invent evidence
    for an unavailable actor or infer whether the physical task succeeded.
    """
    if type(actor_present) is not bool or any(type(value) is not int or value < 0
            for value in (actual_forwards, return_decisions)):
        raise ValueError('Malformed actor presence or forward counts')
    hashes = (start_file_sha256, end_file_sha256, start_state_sha256, end_state_sha256)
    if actor_present:
        if any(not isinstance(value, str) or len(value) != 64 or any(
                char not in '0123456789abcdef' for char in value) for value in hashes):
            raise ValueError('Actual actor hashes are required')
        if start_file_sha256 != end_file_sha256 or start_state_sha256 != end_state_sha256:
            raise ValueError('Actual RETURN actor changed')
        if actual_forwards != return_decisions:
            raise ValueError('RETURN actor calls disagree with delivered decisions')
    elif any(value is not None for value in hashes) or actual_forwards != 0:
        raise ValueError('Feedback RETURN must have no actor hashes or forwards')
    for record in records:
        validate_return_attribution(record, actor_present=actor_present)


class TrialEvidence:
    """One trial's bounded trace and handoff; drain before physical-end/audit.

    Construction and reference loading belong before control starts. Appending
    serializes and copies bytes in memory only. Draining is explicit or occurs
    on context exit, including exceptions; durable-I/O costs remain measured.
    This is a bounded experiment utility, not a hard realtime guarantee.
    """

    def __init__(self, directory: Path, *, max_bytes: int, reference=None):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('A positive fixed evidence budget is required')
        if reference is not None and not isinstance(reference, dict):
            raise ValueError('Handoff reference must be an object')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.max_bytes = max_bytes
        self._trace = []
        self._records = {}
        self._bytes = 0
        self._sealed = False
        self._drained = False
        self.reference = None if reference is None else _json_bytes(reference)
        self.drain_seconds = None
        self.drain_error = None
        self.invalid_error = None
        self.written = []

    @property
    def queued_bytes(self):
        return self._bytes

    @property
    def complete(self):
        return self._drained and self.invalid_error is None and self.drain_error is None

    def _reserve(self, data):
        if self._sealed:
            raise RuntimeError('Evidence buffer is sealed')
        if self.invalid_error is not None:
            raise BufferError(self.invalid_error)
        if self._bytes+len(data) > self.max_bytes:
            self.invalid_error = 'Registered evidence buffer capacity exceeded; trial evidence is incomplete'
            raise BufferError(self.invalid_error)
        self._bytes += len(data)

    def trace(self, text: str):
        if not isinstance(text, str) or not text.endswith('\n'):
            raise ValueError('A complete serialized trace line is required')
        data = text.encode('utf-8')
        self._reserve(data)
        self._trace.append(data)

    def handoff(self, actual: dict):
        if not isinstance(actual, dict):
            raise ValueError('Actual handoff must be an object')
        data = _json_bytes(actual)
        if self.reference is not None and data != self.reference:
            raise ValueError('Actual handoff differs from preloaded reference')
        self.archive_record('return-handoff.json',actual)

    def archive_record(self, name: str, actual: dict):
        """Queue a small named JSON record without allowing path traversal."""
        if (not isinstance(name,str) or Path(name).name != name or not name.endswith('.json')
                or name in self._records or not isinstance(actual,dict)):
            raise ValueError('Invalid or duplicate archival record')
        data = _json_bytes(actual)
        self._reserve(data)
        self._records[name] = data

    def return_physics(self, record: dict, return_actor) -> dict:
        """Emit corrected RETURN metadata without mutating the physical record."""
        emitted = dict(record,control_attribution=return_attribution(return_actor))
        self.trace(json.dumps(emitted,allow_nan=False)+'\n')
        return emitted

    def drain(self):
        if self._drained:
            if self.invalid_error is not None:
                raise BufferError(self.invalid_error)
            return
        if self._sealed:
            raise RuntimeError('Evidence drain already attempted; retained partial files must be inspected')
        self._sealed = True
        began = time.perf_counter()
        try:
            entries = [('trace.jsonl', self._trace)]
            entries.extend((name,[data]) for name,data in self._records.items())
            for name, blocks in entries:
                with (self.directory/name).open('xb') as stream:
                    for block in blocks:
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                self.written.append(name)
            fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            self._drained = True
        except BaseException as exc:
            self.drain_error = repr(exc)
            raise
        finally:
            self.drain_seconds = time.perf_counter()-began
        if self.invalid_error is not None:
            raise BufferError(self.invalid_error)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.drain()
        except BaseException as exc:
            if value is not None:
                raise exc from value
            raise
        return False
