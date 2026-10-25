# Phase 0 baseline and held-out contracts

**Scope:** Summary of the offline source/runtime-bound baseline-freeze and
four-way held-out partition contracts. This is a scoped implementation and
review record. It does not represent a production freeze or complete Phase 0.

## Planner and implementation

- Approved planner: `docs/plans/phase0-baseline-heldout-calibration-20261025.md`,
  final plan commit `f39ab80986e0bbb1db1adc880fa99bc7d7df169c`, SHA-256
  `504484dd9c38446b92faac2837dd56fb0f583643f8bc3dc70ecf21d09afb6fa4`.
- Reviewed implementation head:
  `16aecdc0c9427cdd45c6f7a6f16f1bb07462ca1d`.
- Feature commits: `f45023f77b34a8117f0a3de195df75af2accd8b7`,
  `f5907ed98d64f3d2ea2475425bace5286d6a2c1e`,
  `360c5f074b259b16aee8b2d682ecdbef9e645f1d`,
  `37f3f83351a54f171e0e15d6dba88d12af9943da`, and
  `16aecdc0c9427cdd45c6f7a6f16f1bb07462ca1d`; integration commit:
  `860e2dcabf571a175df30dfcf5ff5c3ed4c3c4d1`.
- The contract follows the MECO track path through
  `meco_track_training.evaluate_meco` → `make_curriculum_env` →
  `CurriculumEnv`/`TensorThreeVsThreeEnv`, its shared low-level simulator and
  source registries. It records eight required configuration artifact IDs and
  SHA-256 digests of supplied content. IDs are stable logical names; the
  freeze does not store file locations, including locations outside the
  checkout. The required IDs are `campaign_assignment`,
  `evaluation_protocol`, `drivetrain_physics`, `gamepiece_rules`,
  `field_layout`, `scenario_registry`, `opponent_schedule`, and
  `sensor_profile`. Compilation and validation fail closed when an ID is
  missing, extra, or has changed content. The source/runtime and field records
  remain explicit inputs; the module does not extract effective values from a
  live campaign.
- The partition contract requires exact `train`, `calibration`, `validation`,
  and `final_test` assignments and rejects duplicates or cross-partition
  overlap for seeds, defender families, layouts, and start conditions. Final
  test is only available to `final_evaluation`. Both contracts declare
  `qualification_eligible: false`.
- Offline implementer checks: 12 focused tests passed; module compilation and
  `git diff --check` passed. No simulator, GPU, device, training, runtime
  calibration, or production artifact was used or created.

## Integrated source hashes

| Path | SHA-256 |
| --- | --- |
| `frc_defense/meco_baseline_freeze.py` | `5c5bc333c01f8a89f7a0e70e5add665621d7458f9a33fea3790b7687b64bd821` |
| `tests/test_meco_baseline_freeze.py` | `043ca040f9b6f86f2b1b1f07b5fd6240438160f03bb5b96d9ccac2162b13567d` |

## Independent review

The independent reviewer approved exact head
`16aecdc0c9427cdd45c6f7a6f16f1bb07462ca1d`. The review confirmed the MECO
`make_curriculum_env` source path, the shared simulator dependency, exact
eight-ID configuration binding, and rejection of missing, extra, or changed
configuration content. The reviewer did not run tests or simulation. This
approval is limited to these offline contracts and does not approve later
runtime evidence or the full Phase 0 plan.

## Limits and remaining work

No production baseline or partition freeze was generated. The caller must
still extract the effective campaign values and supply the correctly
serialized bytes for each required configuration ID; this contract hashes
supplied content but cannot prove that those bytes came from the running
campaign. No calibration, margin review, qualification protocol, match-point
gate, qualification, or promotion is established. The current score semantics
remain scored FUEL piece counts, with official match points unavailable.

No post-ledger dashboard currentness check is claimed here. The API digest pin
and served dashboard will be checked only after integration and service
restart, as described in
`phase0-baseline-dashboard-currentness-plan-20261025.md`.
