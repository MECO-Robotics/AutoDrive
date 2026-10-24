# Matched-reference variance calibration

## Planner scope and acceptance

For a preselected `D1/current`, `D2/cv`, or `D2/multi_model_v1` reference,
collect repeated outcomes on the exact same calibration seeds and explicit
indexed CUDA/HIP device. Freeze policy ID, predictor variant and contract,
seed table, source identity, protocol identity, and normalized runtime. Require
both selected-reference and D0 rows with useful-throughput and all declared
safety metrics. Reject variant mixing, source/runtime/hash drift, incomplete
repeats, and overlap with train, validation, or test. Keep D0 calibration as a
separate safety floor and keep legacy D0 report v2 and qualification protocol
v2/v3 reads intact. Variance evidence informs review only; protocol creation
requires explicit human-reviewed final margins and never derives operational
margins from variance automatically.

## Implementer record

Added `calibrate-matched-reference` and bundle commands in
`frc_defense/meco_qualification.py`, plus a v4 protocol that binds matched
reference report IDs/hashes, seed-table hashes, runtime identity, and an
explicit margin review. The runner reuses the D0 calibration seed partition,
case validation, indexed-device identity, source identity, and normalized
runtime helpers. It evaluates D0 and the selected reference repeatedly on
identical seeds, records all safety metrics, and summarizes both between-seed
and within-seed repeat variance. The bundle and protocol intake validate raw
record coverage, hashes, identities, and disjoint seed pools.

`frc_defense/meco_track_training.py::evaluate_meco` now has an explicit
`calibration_control_policy` opt-in. It accepts only calibration split,
`evaluation_policies=()`, empty checkpoints, controls exactly D0 plus the
selected D1/current or D2/cv/multi_model_v1, a fixed qualification case, no
inference fault, no replay, K=16, and an explicit route-cost weight. It records
the selected control's candidate configuration in its policy manifest. The
calibration runner binds K=16, the training default route-cost weight, and the
12 tick action cadence in both protocol identity and manifest validation. The
existing D0-only path is unchanged; ordinary evaluator validation and
qualification gates remain unchanged. The opt-in supplies paired rows without
loading a learned checkpoint.

Legacy D0 calibration v2 reports and calibrated qualification v2/v3 protocols
retain their original validation and evidence digest paths. New v4 reports use
the expanded evidence digest including the variance-review binding.

Static Python compilation and `git diff --check` are the only checks performed.
No tests, simulator evaluation, replay, GPU workload, dashboard service, or
training output was run or changed. The matched-reference runner currently
executes the complete scenario calibration sequentially in one process; it
does not yet have the D0 shard/resume workflow. That limits interruption
recovery and parallel scheduling and should be addressed before a long
operational campaign. Runtime statistical behavior remains unverified.

## Reviewer

Pending independent review.

## Evidence and runtime status for this tranche

This change adds code paths and intake contracts only. No D0 calibration bundle,
matched-reference report/bundle, margin-review artifact, training output, or
runtime evaluation artifact is present or consumed by this change. Qualification
and promotion remain false until the required evidence is collected and reviewed.
No calibration data was generated, and the evaluator was not run. The CLI
sequence above describes a future evidence collection workflow; it is not
acceptance evidence for this implementation.

Matched-reference raw-row intake also requires exact integer types for both
`repetition` and `seed` before row-key matching. Boolean and floating-point
values are rejected even when Python would consider them equal to an integer.
