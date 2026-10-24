# Phase 4 opponent runtime module

## Three-pass record

- **Planner:** Phase 4 audit defined a deterministic, registry-bound policy
  pool for stationary blockage, direct pursuit, trench interception, feint
  reversal, two reactive policies, and orbit-to-pursuit switching. Policies
  must use only policy-visible sensed state and ordinary bounded commands.
- **Implementer:** `8a31ae2acaba9b0cb5ab842373f2796c271e65bc` adds
  `frc_defense/meco_phase4_opponents.py` and focused tests. It provides stable
  policy IDs and config identity, maps policy IDs to registered factor IDs,
  emits planar velocity requests from supplied sensed state, and fails closed
  on non-finite state, speed limits, or corrupt target-validity metadata. The
  integrated candidate is `f2867715c08e71fd6365b8ddb5ba742e62109fba`.
- **Independent reviewer:** approved exact opponent module commit
  `8a31ae2acaba9b0cb5ab842373f2796c271e65bc` after requesting and reviewing
  fail-closed fixes for NaN command inputs and invalid target-validity values.
  The reviewer verified exact registry pool/factor alignment and no simulator
  truth reads in the pure policy function.

## Evidence and limits

Eight focused tests passed, with Python compilation and `git diff --check`.
No simulator, GPU, replay, or dashboard evaluation was run. The module is not
wired into `CurriculumEnv` or the validation campaign; policy member assignment,
command-unit conversion, realized-event telemetry, held-out family execution,
paired evaluation, and outcome coverage remain outstanding. This is an
approved runtime API only, not executed Phase 4 behavior or qualification.
