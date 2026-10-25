"""Production scorer capture fixtures; intentionally gated on simulator E2E review."""
import pytest
import torch

from frc_defense.meco_simulated_score_ledger import (
    ESTIMAND_SCHEMA, OUTCOME_NAME, PHASE0_PLAN_SHA256, RULES, RULES_ID,
    RULES_SHA256, SOURCE_PLAN_SHA256, MecoScoreEventRecorder,
    estimand_approval_digest,
)
from frc_defense.meco_track_training import _meco_score_ledger_context
from frc_defense.tensor_3v3 import TensorThreeVsThreeEnv


def _approved_fixture(source_sha256):
    return {
        "schema": ESTIMAND_SCHEMA,
        "approval_id": "scorer-path-test-fixture",
        "approved": True,
        "approver": "test-only",
        "approved_at": "2026-10-25T00:00:00Z",
        "source_plan_sha256": SOURCE_PLAN_SHA256,
        "phase0_plan_sha256": PHASE0_PLAN_SHA256,
        "simulator_source_sha256": source_sha256,
        "rules_id": RULES_ID,
        "rules_sha256": RULES_SHA256,
        "primary_estimand": OUTCOME_NAME,
        "formula": "sum score_units by alliance over accepted FUEL score-award events",
        "unit": RULES["unit"],
        "included_outcomes": RULES["included_outcomes"],
        "excluded_outcomes": RULES["excluded_outcomes"],
        "interpretation_limits": "Test fixture only; not official points or causal contribution.",
        "intended_use": "Unit fixture only.",
    }


@pytest.mark.skip(reason=(
    "Run only with the required seeded simulator E2E evaluation and fresh dashboard-currentness evidence."))
@pytest.mark.parametrize("vectorized", [True, False], ids=["vectorized", "robot-loop"])
def test_update_fuel_actions_captures_award_and_advances_generation(vectorized):
    env = TensorThreeVsThreeEnv(num_envs=1, device="cpu", seed=91,
        fuel_count=96, randomize=False, perception_dropout=0,
        position_noise=0, velocity_noise=0)
    env._vectorized_scoring_enabled = vectorized
    env.piece_active.zero_()
    env.piece_owner.fill_(-1)
    env.piece_active[0, 0] = True
    env.piece_owner[0, 0] = 0
    hub = env.hub_centers[0]
    env.sim.pose[0, 0] = torch.stack((hub[0] - 0.5, hub[1], hub.new_zeros(())))
    env.sim.velocity[0, 0].zero_()
    env.match_elapsed.fill_(env.dt)
    env.steps.fill_(1)
    env.next_score.zero_()
    env.hub_active.fill_(True)

    source_sha256 = "a" * 64
    approval = _approved_fixture(source_sha256)
    context = _meco_score_ledger_context(env, episode_id="fixture", seed=91,
        scenario_id="pickup_simple", source_identity_sha256=source_sha256)
    recorder = MecoScoreEventRecorder(context, approval,
                                      estimand_approval_digest(approval))
    env._meco_score_event_recorder = recorder

    env._update_fuel(torch.ones(1, dtype=torch.bool),
                     torch.tensor([[True, False, False, False, False, False]]))

    assert len(recorder.events) == 1
    assert recorder.events[0]["piece_slot"] == 0
    assert recorder.events[0]["piece_generation"] == 0
    assert recorder.events[0]["source_site"].endswith(
        "vectorized_score_mutation" if vectorized else "robot_loop_score_mutation")
    assert env._meco_piece_generation[0, 0].item() == 1
