# Phase 4 legal full-team start profiles — scoped summary

## Planner scope

The scoped Phase 4 review identified that the registry-backed
`start.legal_team_poses` assignment was planned but not wired through validation
dispatch and curriculum reset. The accepted contract was to apply the same
protocol-bound assignment consistently to matched treatment episodes, retain
the ordinary-reset control, and prevent perception or observation from seeing
the pre-assignment pose. The six robot poses must pass field, obstacle, and
pairwise clearance validation. This tranche must not infer authoritative
outcomes or qualification from pose validity.

## Implementation

Implementation commit: `28f906961115adffb11a3e7e11e4188e4660dcd9`.

Changed paths:

- `frc_defense/architecture_scenarios.py`
- `frc_defense/curriculum_scenarios.py`
- `frc_defense/meco_ablation_matrix.py`
- `frc_defense/meco_phase4_campaign_plan.py`
- `frc_defense/meco_phase4_context.py`
- `frc_defense/meco_phase4_runtime_adapter.py`
- `frc_defense/meco_track_training.py`
- `frc_defense/meco_validation_campaign_runner.py`
- `frc_defense/tensor_3v3.py`
- `scripts/run_meco_ablation_matrix.py`
- `tests/test_meco_phase4_campaign_plan.py`
- `tests/test_meco_phase4_context.py`
- `tests/test_meco_phase4_runtime_adapter_contract.py`

The implementation binds the registered profile to the exact validation
protocol, case, category, and seed assignment. Matched control and treatment
matrix arms are checked against frozen checkpoint hashes, source identity, and
execution device; per-policy reports and episode rows must retain those same
bindings. Curriculum environment construction defers its initial perception
until assignment is attached. Reset applies the full-team profile before the
final perception refresh/observation, clears velocity and stale planner/sensor
state, and rejects pose sets that fail six-robot field, solid, or pairwise
clearance checks. Records preserve profile/config digests while leaving
`outcome_by_start_state` null and qualification false.

## Independent review

The exact implementation commit `28f906961115adffb11a3e7e11e4188e4660dcd9`
was approved for this scoped tranche. Review findings required the dispatcher
to bind both matrix arms and each per-policy report to identical checkpoint,
source, and device provenance, and required constructor-time perception to be
deferred until the profile assignment exists. Those checks are represented in
the approved commit and focused CPU contracts.

## Checks

- `python -m pytest -q tests/test_meco_phase4_campaign_plan.py::test_phase4_start_profile_dispatch_pairs_all_policies_and_exact_seeds tests/test_meco_phase4_campaign_plan.py::test_curriculum_constructor_defers_initial_perception_until_profile_attachment tests/test_meco_phase4_campaign_plan.py::test_phase4_dispatch_binding_rejects_report_drift`: **5 passed**.
- After strengthening the constructor/reset-order assertion, the focused static
  contract was rerun alone: **1 passed**.
- `py_compile` passed for the changed campaign runner, evaluator, curriculum
  scenario, environment, and contract-test modules; `git diff --check` passed.

## Limits

This is reviewed implementation and static/mock contract evidence only. No
simulator evaluation, GPU/HIP run, training, campaign execution, replay, or
dashboard service validation was performed. The static constructor contract
does not establish runtime perception ordering in a live simulator. Six-pose
legal-start rate and outcome-by-start-state remain unavailable/null; simulator
behavior is unverified and qualification eligibility remains false.
