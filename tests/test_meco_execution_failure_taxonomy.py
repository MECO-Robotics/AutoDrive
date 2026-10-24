import pytest

from frc_defense.meco_track_training import (
    _evaluation_action_was_remasked,
    _evaluation_execution_failure_reason,
)


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


def test_action_7_inference_fallback_with_only_action_7_ticks_is_not_remasked():
    selected_action = 7
    executed_action_trace = [7] * 12
    remasked = _evaluation_action_was_remasked(selected_action,
        executed_action_trace)

    assert remasked is False
    assert _evaluation_execution_failure_reason(
        action_was_remasked=remasked, native_preempted=False,
        selected_target_lost=False, policy_fallback=True,
        fallback_reason="policy_inference_fallback") == "policy_inference_fallback"


def test_action_7_trace_is_remasked_only_when_an_executed_tick_differs():
    assert _evaluation_action_was_remasked(7, [7] * 11 + [6]) is True


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


def test_target_loss_and_native_preemption_precede_remask_and_fallback():
    assert _evaluation_execution_failure_reason(
        action_was_remasked=True, native_preempted=True,
        selected_target_lost=True, policy_fallback=True,
        fallback_reason="policy_inference_fallback") == "selected_target_lost"
    assert _evaluation_execution_failure_reason(
        action_was_remasked=True, native_preempted=True,
        selected_target_lost=False, policy_fallback=True,
        fallback_reason="policy_inference_fallback") == (
            "native_behavior_preempted_selected_goal")


def test_failure_classifier_rejects_clean_execution():
    with pytest.raises(ValueError, match="without a failure"):
        _evaluation_execution_failure_reason(
            action_was_remasked=False, native_preempted=False,
            selected_target_lost=False, policy_fallback=False,
            fallback_reason=None)
