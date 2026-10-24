import pytest

from frc_defense.meco_track_training import _evaluation_execution_failure_reason


def test_inference_fallback_is_not_reported_as_action_remasking():
    assert _evaluation_execution_failure_reason(
        action_was_remasked=False, native_preempted=False,
        selected_target_lost=False, policy_fallback=True,
        fallback_reason="policy_inference_fallback") == "policy_inference_fallback"


def test_nonfinite_decision_has_explicit_reason():
    assert _evaluation_execution_failure_reason(
        action_was_remasked=False, native_preempted=False,
        selected_target_lost=False, policy_fallback=True,
        fallback_reason="nonfinite_policy_decision") == "nonfinite_policy_decision"


@pytest.mark.parametrize(("kwargs", "expected"), [
    ({"action_was_remasked": True, "native_preempted": False,
      "selected_target_lost": False, "policy_fallback": True,
      "fallback_reason": "policy_inference_fallback"}, "action_was_remasked"),
    ({"action_was_remasked": False, "native_preempted": True,
      "selected_target_lost": False, "policy_fallback": False,
      "fallback_reason": None}, "native_behavior_preempted_selected_goal"),
    ({"action_was_remasked": False, "native_preempted": False,
      "selected_target_lost": True, "policy_fallback": False,
      "fallback_reason": None}, "selected_target_lost"),
])
def test_observed_execution_faults_keep_specific_reasons(kwargs, expected):
    assert _evaluation_execution_failure_reason(**kwargs) == expected


def test_failure_classifier_rejects_clean_execution():
    with pytest.raises(ValueError, match="without a failure"):
        _evaluation_execution_failure_reason(
            action_was_remasked=False, native_preempted=False,
            selected_target_lost=False, policy_fallback=False,
            fallback_reason=None)
