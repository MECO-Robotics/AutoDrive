import hashlib
import json
from pathlib import Path

import pytest

from frc_defense.runtime_evidence import (
    TrialEvidence, load_bound_reference, return_attribution, validate_return_attribution,
    validate_return_run,
)


def test_buffer_preserves_bytes_order_and_copies_handoff(tmp_path):
    directory = tmp_path/'trial'
    actual = {'tick':20,'state':[1.,2.]}
    reference = {'tick':20,'state':[1.,2.]}
    with TrialEvidence(directory, max_bytes=1024, reference=reference) as evidence:
        evidence.trace('{"tick": 0}\n')
        evidence.trace('{"tick": 5}\n')
        evidence.handoff(actual)
        actual['state'][0] = 99.
        assert list(directory.iterdir()) == []
    assert (directory/'trace.jsonl').read_bytes() == b'{"tick": 0}\n{"tick": 5}\n'
    assert json.loads((directory/'return-handoff.json').read_text()) == reference
    assert evidence.drain_seconds is not None and evidence.drain_error is None
    assert evidence.complete
    evidence.drain()  # A successful explicit drain plus context exit is harmless.
    with pytest.raises(RuntimeError):
        evidence.trace('{}\n')


def test_exception_still_drains_existing_evidence(tmp_path):
    evidence = TrialEvidence(tmp_path/'trial', max_bytes=1024)
    with pytest.raises(RuntimeError, match='original trial failure'):
        with evidence:
            evidence.trace('{}\n')
            raise RuntimeError('original trial failure')
    assert (evidence.directory/'trace.jsonl').read_text() == '{}\n'


def test_overflow_retains_accepted_prefix_and_fails(tmp_path):
    evidence = TrialEvidence(tmp_path/'trial', max_bytes=3)
    with pytest.raises(BufferError):
        with evidence:
            evidence.trace('{}\n')
            evidence.trace('{}\n')
    assert evidence.queued_bytes == 3
    assert (evidence.directory/'trace.jsonl').read_bytes() == b'{}\n'
    assert evidence.invalid_error and not evidence.complete


def test_caught_overflow_cannot_be_reported_complete(tmp_path):
    evidence = TrialEvidence(tmp_path/'trial',max_bytes=3)
    evidence.trace('{}\n')
    with pytest.raises(BufferError):
        evidence.trace('{}\n')
    with pytest.raises(BufferError):
        evidence.drain()
    assert (evidence.directory/'trace.jsonl').read_bytes() == b'{}\n'
    assert not evidence.complete and evidence.invalid_error
    with pytest.raises(BufferError):
        evidence.drain()


def test_duplicate_and_mismatched_handoff_rejected(tmp_path):
    with TrialEvidence(tmp_path/'a',max_bytes=1024,reference={'tick':20}) as evidence:
        with pytest.raises(ValueError):
            evidence.handoff({'tick':21})
        evidence.handoff({'tick':20})
        with pytest.raises(ValueError):
            evidence.handoff({'tick':20})


def test_return_emitter_and_named_records_do_not_change_input(tmp_path):
    physical = {'task_phase':'return','position':[1.,2.],'velocity':[.1,.2]}
    with TrialEvidence(tmp_path/'trial',max_bytes=1024) as evidence:
        emitted = evidence.return_physics(physical,None)
        evidence.archive_record('first60-physical.json',{'tick':100})
        assert 'control_attribution' not in physical
        assert emitted['position'] == physical['position'] and emitted['velocity'] == physical['velocity']
        validate_return_attribution(emitted,actor_present=False)
        for name in ['../escape.json','/absolute.json','trace.jsonl','first60-physical.json']:
            with pytest.raises(ValueError):
                evidence.archive_record(name,{})


def test_bound_reference_and_malformed_reference(tmp_path):
    reference = tmp_path/'reference.json'
    reference.write_text('{"tick":20}\n')
    assert load_bound_reference(reference,hashlib.sha256(reference.read_bytes()).hexdigest()) == {'tick':20}
    with pytest.raises(ValueError):
        load_bound_reference(reference,'0'*64)
    reference.write_text('[]\n')
    with pytest.raises(ValueError):
        load_bound_reference(reference,hashlib.sha256(reference.read_bytes()).hexdigest())


def test_fsync_failure_remains_explicit_and_cannot_be_retried(tmp_path, monkeypatch):
    evidence = TrialEvidence(tmp_path/'trial', max_bytes=1024)
    evidence.trace('{}\n')
    def fail(_):
        raise OSError('intentional fsync failure')
    monkeypatch.setattr('frc_defense.runtime_evidence.os.fsync',fail)
    with pytest.raises(OSError,match='intentional fsync failure'):
        evidence.drain()
    assert evidence.drain_error and evidence.drain_seconds is not None
    with pytest.raises(RuntimeError):
        evidence.drain()


def test_write_failure_keeps_original_exception_context(tmp_path, monkeypatch):
    evidence = TrialEvidence(tmp_path/'trial', max_bytes=1024)
    original = Path.open
    def fail(path, *args, **kwargs):
        if path == evidence.directory/'trace.jsonl':
            raise OSError('intentional write failure')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',fail)
    with pytest.raises(OSError) as caught:
        with evidence:
            evidence.trace('{}\n')
            raise RuntimeError('original failure')
    assert isinstance(caught.value.__cause__,RuntimeError)
    assert 'original failure' in str(caught.value.__cause__)


@pytest.mark.parametrize('actor', [None, object()])
def test_attribution_uses_actual_actor(actor):
    row = {'control_attribution':return_attribution(actor)}
    validate_return_attribution(row,actor_present=actor is not None)
    with pytest.raises(ValueError):
        validate_return_attribution(row,actor_present=actor is None)


@pytest.mark.parametrize('flag', [0,1,'true',None])
def test_nonboolean_attribution_rejected(flag):
    row = {'control_attribution':{'mode':'conditional-return-neural','navigation_input_to_NN':flag}}
    with pytest.raises(ValueError):
        validate_return_attribution(row,actor_present=True)


@pytest.mark.parametrize('actor', [False,True])
def test_actual_run_hashes_and_forward_counts(actor):
    row = {'control_attribution':return_attribution(object() if actor else None)}
    args = dict(actor_present=actor,start_file_sha256='a'*64 if actor else None,
        end_file_sha256='a'*64 if actor else None,start_state_sha256='b'*64 if actor else None,
        end_state_sha256='b'*64 if actor else None,actual_forwards=3 if actor else 0,return_decisions=3)
    validate_return_run([row]*15,**args)
    for field in ['end_file_sha256','end_state_sha256','actual_forwards']:
        bad = dict(args)
        bad[field] = 2 if field=='actual_forwards' else ('c'*64 if actor else 'a'*64)
        with pytest.raises(ValueError):
            validate_return_run([row]*15,**bad)
