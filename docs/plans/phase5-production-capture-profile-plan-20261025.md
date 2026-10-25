# Phase 5 production-capture profile plan

## Planner pass

**Scope:** add a checkpoint-bound profiling mode that consumes the same exact
production-request captures already used by ONNX export and parity. This is a
bounded Phase 5 software gate: it replaces synthetic scorer inputs with
recorded production tensors for the scorer microbenchmark and emits auditable
checkpoint, capture, source, and input bindings. It does not claim deployment
qualification.

**Planning base:** `main` at `25d00688f14bdcf5863550e34e2bb6d8d411d18a`.
The root worktree contains an unrelated modified generated HIP source file;
this plan is isolated in a dedicated worktree and does not touch it.

## Evidence from the current implementation

- The full plan requires ONNX export, actual Hailo compiler/operator and
  memory checks where available, and measurement of CPU fallback, transfer
  overhead, inference age, AD* route computation, action arbitration, and
  effective score-to-actuation latency (plan §Phase 5, line 108).
- `docs/MECO_DEPLOYMENT_PROFILE.md` explicitly classifies the current profile
  as synthetic scorer-only evidence. It says inputs are fixed-shape synthetic
  tensors, transfer is excluded, the age field repeats scorer timing, and
  route feasibility/action arbitration and actuation are not measured.
- `frc_defense.meco_request_capture.load_capture` and
  `validate_capture_inputs` already load and validate captured tensor content,
  canonical order, contract, and request provenance. `scripts/export_meco_onnx.py`
  and `scripts/verify_meco_onnx_parity.py` require exactly one capture per
  track and bind it to a selected checkpoint and export manifest.
- In `frc_defense/meco_deployment.py`, `benchmark()` always creates inputs with
  `dummy_inputs(track, ...)`; `profile_report()` has no capture or export
  manifest input. It benchmarks CPU plus each visible accelerator, but the
  returned report calls its pipeline `synthetic_raw_scorer_argmax_ms` and
  declares `synthetic_diagnostic_only`. `scripts/profile_meco_deployment.py`
  exposes no production-capture options.
- The status ledger records that production-capture ONNX parity code exists,
  while no trained-checkpoint production capture, Hailo/device qualification,
  or deployment qualification is claimed. Captures therefore cannot be
  presumed to exist or match future trained checkpoints.

## Proposed implementation tranche

Extend only the existing scorer microbenchmark to accept an optional
`--production-capture-dir` and, when enabled, profile the exact captured
per-track tensors rather than `dummy_inputs`.

1. In `scripts/profile_meco_deployment.py`, add an explicit capture-directory
   option and require the complete five-track set for this evidence mode.
   Require exactly one canonical `<TRACK>-*.npz` capture for A1, A2, A3, C1,
   and C2; reject missing, duplicate, or unexpected track captures.
2. In `frc_defense/meco_deployment.py`, reuse `load_capture` and
   `validate_capture_inputs`; validate capture/checkpoint equality for track,
   checkpoint SHA-256, protocol ID, source snapshot digest, and current input
   contract. Validate canonical names, tensor order, dtype, contiguity,
   finiteness, and dimensions before timing. Do not silently fall back to
   synthetic inputs if any capture fails.
3. Feed captured tensors through the existing scorer benchmark on CPU and each
   explicitly available ROCm device. Keep device transfer outside the timed
   scorer interval unless a distinct transfer measurement is added and named
   accurately. Use a separate report schema/evidence class for capture-backed
   scorer microbenchmarks; preserve the existing synthetic mode and labels.
4. Bind each result to the capture's aggregate content and metadata digests,
   request/snapshot/generation identity, contract digest, input names/shapes/
   dtypes and content digest, plus checkpoint SHA-256, run/update/protocol,
   source snapshot digest, benchmark source identity, device runtime identity,
   iteration/warmup counts, and captured-versus-synthetic input mode.
   A changed/replaced capture or checkpoint must invalidate the run before
   benchmarking.
5. Update `docs/MECO_DEPLOYMENT_PROFILE.md` with the command, report fields,
   capture creation prerequisites, and strict interpretation: this profiles
   trained scorers on recorded inputs only. It is not transfer, freshness,
   complete decision-pipeline, Hailo, actuation, or qualification evidence.

Recommended CLI shape:

```bash
python scripts/profile_meco_deployment.py \
  --checkpoint-root checkpoints/meco-five \
  --production-capture-dir evaluations/meco-production-captures \
  --output evaluations/meco-deployment/production-capture-profile.json
```

Synthetic mode remains an explicit diagnostic and does not gain capture
provenance by inference. Capture mode should fail closed if PyTorch cannot load
the selected checkpoint, a source/protocol/checkpoint binding differs, any
track is absent, or an accelerator requested for the run is unavailable.

## Three-pass execution and review gates

### Planner

This artifact defines the scope, known callsites, evidence contract, and limits.
The planner does not change runtime code, run tests, exercise a simulator,
access GPUs, or interrupt training.

### Implementer

Implement in an isolated worktree. Keep synthetic profiling behavior
backward-compatible. Add focused contract coverage for all-five completeness,
duplicate/missing captures, malformed or mismatched checkpoint/source/protocol
bindings, digest replacement, invalid tensors, and report provenance. Do not
claim test coverage for real device timing. `AGENTS.md` requires representative
end-to-end simulator evaluation and replay inspection for any test run that
checks simulated behavior; this tranche's contract checks must remain
non-simulator checks. Do not run tests while the user/developer instruction
prohibits tests.

### Independent reviewer

Review the exact implementation head and confirm: all five tracks are required
in capture mode; captures are validated against selected checkpoints before
timing; report content digests match the consumed tensors; synthetic and
capture-backed evidence are distinguishable; no fallback can convert a failed
capture run to synthetic success; and the documentation makes no qualification
or full-pipeline claim. Corrections get separate commits and exact-head
re-review before integration.

## Runtime artifact and provenance contract

When trained checkpoints and matching production captures become available,
the output should be one immutable JSON report with:

- schema and `evidence_class` identifying a production-capture scorer profile;
- exact five-track coverage and a top-level pass/fail status for artifact
  validation (timing records can still be marked unavailable per device);
- per-track checkpoint identity: file SHA-256, run ID, update, protocol,
  source snapshot and digest;
- per-track capture path, content/metadata digests, input-contract digest,
  request sequence, generation ID, snapshot ID, candidate-mask digest and
  legal-pose-order digest when applicable;
- exact input names/order, shapes, dtypes and content digest;
- measured scorer latency distributions, peak memory and runtime/device
  versions, with transfer excluded and clearly declared;
- profile implementation source identity and exact run configuration.

Store the report next to the capture and checkpoint bundle and include its
digest in the project evidence/provenance ledger. Never overwrite a prior
report for a different checkpoint or capture set.

## Dependencies and ordering

This tranche can be implemented and contract-reviewed without GPU/Hailo access.
Runtime evidence depends on (1) a completed five-track checkpoint bundle from
the intended source/protocol, (2) one fresh production request capture per
track produced through the actual adapter with capture explicitly enabled,
and (3) a later deliberate profile run with checkpoint and capture hashes
rechecked. Existing uninterrupted r4 runs use an older source and have no
current evidence that matching captures exist; do not treat their checkpoints
as compatible or change/stop that run to produce these artifacts.

The next physical deployment gates remain separate: actual WX9100 parity and
profile if that is the selected inference target; Hailo operator compilation,
device memory fit, and on-device latency on the real target; measured host to
device transfer; production request-age under load; full AD* and candidate
construction, feasibility, arbitration, simulator replay, and robot
score-to-actuation timing. Phase 4 generalization and model qualification also
remain outstanding. No profiler schema or green synthetic check can satisfy
those evidence requirements.

## Acceptance criteria

- `--production-capture-dir` selects strict capture-backed scorer profiling;
  absent option preserves existing synthetic diagnostics.
- A valid run binds all five exact captures to their selected checkpoints,
  validates them before profiling, records consumed input digests and runtime
  provenance, and labels timed stages accurately.
- Any incomplete/malformed/mutated/mismatched capture set or any checkpoint
  identity mismatch fails the report and exits nonzero; no synthetic fallback.
- Documentation and ledger classify results as scorer-only evidence and keep
  Hailo/deployment qualification false until the corresponding target hardware
  and end-to-end gates are run.
