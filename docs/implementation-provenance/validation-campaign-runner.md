# Resumable MECO validation campaign runner: three-pass summary

## Planner pass

The validation planner already emits the exact 28 validation-only jobs, but
there was no resumable execution receipt or cross-scenario aggregation. The
planner scoped a validation-only runner that binds all five checkpoints,
configuration hashes, source/runtime identity, protocol, exact seeds and
condition coverage; validates replay/report provenance; and seals completed
jobs immutably. Unsafe partial output must fail closed. Aggregation must retain
scenario identity and avoid an unsupported pooled estimate. Final-test seeds
and qualification claims are excluded under the current provisional protocol.

## Implementer pass

Added `frc_defense/meco_validation_campaign_runner.py` and run/summarize CLI
entrypoints. The runner verifies the exact 28-job plan, protocol and checkpoint
digests, all-track source/config bindings, and indexed accelerator selection.
It uses per-device and per-campaign exclusive locks, immutable job inputs,
digest-checked completion seals, safe resume, and fail-closed handling for
partial outputs. Complete matrices are validated against exact condition,
policy/category/seed and replay coverage before sealing. The summarizer
requires the exact completed job set, verifies seals/artifacts, retains
scenario/category identity, and emits per-scenario diagnostics without pooled
estimates.

The independent replay review found that the simulator writes identity under
`replay_provenance`; this was corrected to bind replay pointer and evaluation
row digest plus source, protocol, checkpoint, route, outcome, and safety
digests. A producer-shaped replay fixture includes a digest-tampering check.

Implementer commits: `4ad9291ed652cf02ccae3958fc024afa9f90fb51` and
`6c1d5eb04acb39b192f1b491f753ca68d338096c`. Candidate integration commits:
`dec8db8d5` and `99dd35995`.

## Independent reviewer pass

The reviewer approved exact final implementer HEAD
`6c1d5eb04acb39b192f1b491f753ca68d338096c` after finding and requiring the
replay provenance schema fix. Review checked the diagnostic-only gate, exact
coverage, immutable seals, lock behavior, safe resume, fail-closed partial
outputs, replay provenance, and scenario-preserving aggregation.

## Verification and limits

Ten focused planner/runner tests passed; Python compilation and
`git diff --check` passed. Tests mock matrix execution. No simulator, GPU,
dashboard, replay, or final-test operation was run. The runner and aggregator
are implemented, but no validation campaign has been executed and there are no
matrix results, cross-scenario outcome estimates, qualification outcomes, or
fresh replay evidence.
