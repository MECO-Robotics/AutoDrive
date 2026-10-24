# Phase 4 reactive opponent validation wiring — scoped summary

## Planner gap

The Phase 4 deterministic opponent pool in `frc_defense/meco_phase4_opponents.py` exposed policy command helpers, but the helper-only API was not connected to validation episode scheduling and evaluation. The scoped pass added a protocol-bound schedule, a matrix CLI condition selector, dispatcher-to-evaluator assignment threading, and policy-visible sensed-state command inputs. Both paired matrix arms are restricted to `architecture_matched_cv_k16` so default feint/abrupt factor conditions cannot overlap this treatment.

## Implementation

Implementation commit: `49c8ba7d15c507c1208600daf3b3fa3b4312a15e`.

Changed paths:

- `frc_defense/curriculum_scenarios.py`
- `frc_defense/meco_ablation_matrix.py`
- `frc_defense/meco_phase4_campaign_plan.py`
- `frc_defense/meco_phase4_opponents.py`
- `frc_defense/meco_phase4_runtime_adapter.py`
- `frc_defense/meco_track_training.py`
- `frc_defense/meco_validation_campaign_runner.py`
- `scripts/run_meco_ablation_matrix.py`
- `tests/test_meco_phase4_campaign_plan.py`
- `tests/test_meco_phase4_opponents.py`

## Independent review

Two independent reviewers approved the revised source at exact HEAD `983183665fc32fc4f8d6d501a0258ebb7e25a6a0`, immediately before implementation commit `49c8ba7d1`:

- `/root/phase4_pair_contact_review`: approved the paired coverage, exact architecture-only condition, pressure-slot scope, policy/control/category/seed pairing, and fail-closed null outcome/qualification fields; cited 19 focused tests passing and no simulator/GPU validation.
- `/root/phase4_dispatch_review`: approved condition selection in both arms, explicit factor identity flowing from the planned cell through schedule compilation/validation, separate policy-family identity, and all seven policy IDs covered by the assignment hash contract; no code blocker.

## Static contract evidence

The focused command `PYTHONPATH=. python -m pytest tests/test_meco_phase4_opponents.py tests/test_meco_phase4_campaign_plan.py -q` passed **19 tests**. `py_compile` passed for the matrix, campaign planner, opponent policy module, evaluator, campaign runner, and matrix CLI. `git diff --check` passed. The dispatcher contract checks exact architecture-only condition selection in both arms and validates paired artifact coverage.

## Limits

This tranche is diagnostic only. Configured pressure slots are slot 0 for levels 1–4 and slots 0/1 for level 5; slot 2 is out of scope. No simulator behavior validation or replay was run, so assignment-to-slot application and resulting behavior remain unverified. Family-specific outcomes, including interception rate and paired outcome by family, remain null/unavailable. Qualification eligibility is false. No GPU, training, dashboard, or qualification action was performed.
