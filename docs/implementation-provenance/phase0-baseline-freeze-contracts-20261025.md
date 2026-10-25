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
- The contract follows the MECO track path through
  `meco_track_training.evaluate_meco` → `make_curriculum_env` →
  `CurriculumEnv`/`TensorThreeVsThreeEnv`, its shared low-level simulator and
  source registries. It records eight required configuration artifact IDs and
  SHA-256 digests of supplied content. IDs are stable logical names; the
  freeze does not store file locations, including locations outside the
  checkout. Compilation and validation fail closed when an ID is missing,
  extra, or has changed content. The source/runtime and field records remain
  explicit inputs; the module does not extract effective values from a live
  campaign.
- The partition contract requires exact `train`, `calibration`, `validation`,
  and `final_test` assignments and rejects duplicates or cross-partition
  overlap for seeds, defender families, layouts, and start conditions. Final
  test is only available to `final_evaluation`. Both contracts declare
  `qualification_eligible: false`.
- Offline implementer checks: 12 focused tests passed; module compilation and
  `git diff --check` passed. No simulator, GPU, device, training, runtime
  calibration, or production artifact was used or created.

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
