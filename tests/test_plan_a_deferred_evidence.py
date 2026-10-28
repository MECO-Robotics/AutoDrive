from dataclasses import dataclass
import inspect
import torch
import pytest

from frc_defense import meco_track_training as training


@dataclass(frozen=True)
class Support:
    support_digest: str


def test_action_digest_and_forecast_are_one_payload_and_decode_exactly():
    action = torch.arange(13, dtype=torch.float64)
    digest = torch.arange(32, dtype=torch.uint8)
    forecast = torch.tensor([.25, 3.5, 1., 1., 0., 0., .75], dtype=torch.float32)
    packed = training._pack_plan_a_decision_evidence(action, digest, forecast)
    assert packed.shape == (52,)
    assert torch.equal(packed[:13], action)
    prepared = {'support': Support(''), 'support_digest_scheme': 'tree-sha256-v2', 'prediction': {'_device_report': forecast}}
    training._materialize_plan_a_decision_evidence(prepared, packed.tolist())
    assert prepared['support'].support_digest == bytes(range(32)).hex()
    assert prepared['prediction']['horizon_min_s'] == .25
    assert prepared['prediction']['horizon_max_s'] == 3.5
    assert prepared['prediction']['used_cv'] is True
    assert prepared['prediction']['observed_age_s'] == .75


def test_compact_evidence_rejects_bad_shapes_and_keeps_other_tracks_unchanged():
    with pytest.raises(ValueError, match='invalid Plan A compact'):
        training._pack_plan_a_decision_evidence(torch.zeros(13),
            torch.zeros(31, dtype=torch.uint8), torch.zeros(7))
    training._materialize_plan_a_decision_evidence({'candidate_set': None}, None)


def test_prepare_does_not_materialize_evidence_before_handoff():
    source = inspect.getsource(training._prepare_plan_a_decision)
    assert 'plan_a_support_digest(' not in source
    assert 'support_tree_digest_device(' in source
    assert 'support_tree_digest_device(' in source


@pytest.mark.parametrize("bad", [1.5, -1., 256., float("nan"), float("inf")])
def test_compact_digest_rejects_malformed_bytes(bad):
    values = [0.] * 52
    values[13] = bad
    prepared = {'support': Support(''), 'support_digest_scheme': 'tree-sha256-v2',
                'prediction': {'_device_report': torch.zeros(7)}}
    with pytest.raises(ValueError, match='invalid Plan A compact digest bytes'):
        training._materialize_plan_a_decision_evidence(prepared, values)
    assert prepared['support'].support_digest == ''
