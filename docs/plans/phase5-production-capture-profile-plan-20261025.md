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
  exposes no production-capture options. The repository already has
  `frc_defense/meco_gpu_attestation.py::verify_post_lease_wx9100` and the
  pinned `qualifications/meco/host-gpu-inventory-20261024.json`; the profile
  must use those physical-identity checks rather than trust logical device
  names or enumerate visible accelerators.
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
3. Acquire an exclusive lease for exactly `cuda:0` and `cuda:1`, then call
   `verify_post_lease_wx9100` for that exact pair before any checkpoint/model
   profiling. Require both physical identities from the pinned inventory:
   `cuda:0` = WX9100 at BDF `0000:19:00.0`, PCI ID `1002:6861`, HIP UUID
   `30323133-6632-6261-3734-313632316134`; `cuda:1` = WX9100 at BDF
   `0000:67:00.0`, PCI ID `1002:6861`, HIP UUID
   `30323133-6661-3836-3737-346334386134`. Also preserve the attestor's
   explicit P600 exclusion. If the lease, either device, identity property,
   attestation tool, or pinned inventory check is unavailable or mismatched,
   fail closed; do not profile the remaining single card, CPU, or another
   visible accelerator. Run every track on both attested WX9100s. CPU fallback
   profiling is outside this tranche; it may be a separate, explicitly
   optional deployment study with its own evidence class and scope.
4. Keep captured-input host-to-device transfer outside the timed scorer
   interval and label that exclusion. Do not report transfer, inference age,
   or end-to-end latency from this scorer-only measurement. Use a separate
   report schema/evidence class for capture-backed scorer microbenchmarks;
   preserve the existing synthetic mode and labels.
5. Bind each result to the capture's aggregate content and metadata digests,
   request/snapshot/generation identity, contract digest, input names/shapes/
   dtypes and content digest, plus checkpoint SHA-256, run/update/protocol,
   source snapshot digest, profiler source identity, physical device identity,
   runtime identity, iteration/warmup counts, and input mode. Define profiler
   source identity as the Git commit SHA plus a clean working-tree assertion;
   reject a dirty checkout before profiling. Record this independently from
   the checkpoint's training source snapshot and digest. Do not conflate or
   compare the current profiler commit with the historical source snapshot
   bound into a checkpoint. A changed/replaced capture or checkpoint must
   invalidate the run before benchmarking.
6. Update `docs/MECO_DEPLOYMENT_PROFILE.md` with the command, report fields,
   capture creation prerequisites, and strict interpretation: this profiles
   trained scorers on recorded inputs only. It is not transfer, freshness,
   complete decision-pipeline, Hailo, actuation, or qualification evidence.

Recommended CLI shape:

```bash
python scripts/profile_meco_deployment.py \
  --checkpoint-root checkpoints/meco-five \
  --production-capture-dir evaluations/meco-production-captures \
  --device cuda:0 --device cuda:1 \
  --output evaluations/meco-deployment/production-capture-profile.json
```

Synthetic mode remains an explicit diagnostic and does not gain capture
provenance by inference. Production-capture mode requires both explicit device
arguments as the exact pair `cuda:0`, `cuda:1`; under an exclusive lease it
must attest both against the pinned WX9100 inventory with
`verify_post_lease_wx9100`. It fails closed if PyTorch cannot load the selected
checkpoint, the current profiler checkout is dirty, a source/protocol/
checkpoint binding differs, any track is absent, either WX9100 is unavailable,
or either physical identity differs. It never substitutes CPU or one visible
GPU for the required pair.

## Three-pass execution and review gates

### Planner

This artifact defines the scope, known callsites, evidence contract, and limits.
The planner does not change runtime code, run tests, exercise a simulator,
access GPUs, or interrupt training.

### Implementer

Implement in an isolated worktree. Keep synthetic profiling behavior
backward-compatible. Add focused contract coverage for all-five completeness,
duplicate/missing captures, malformed or mismatched checkpoint/source/protocol
bindings, digest replacement, invalid tensors, both-device requirement,
physical identity mismatch/unavailability, dirty profiler checkout, and
separate profiler-commit versus checkpoint-source provenance. Do not claim
test coverage for real device timing. `AGENTS.md` requires representative
end-to-end simulator evaluation and replay inspection for any test run that
checks simulated behavior; this tranche's contract checks must remain
non-simulator checks. Do not run tests while the user/developer instruction
prohibits tests.

### Independent reviewer

Review the exact implementation head and confirm: all five tracks are required
in capture mode; captures are validated against selected checkpoints before
timing; exactly the two pinned WX9100s are leased and physically attested;
failure of either device cannot fall back to CPU or a lone/other accelerator;
report content digests match consumed tensors; profiler commit/clean-tree
identity is separate from checkpoint training-source identity; synthetic and
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
- measured scorer latency distributions and peak memory separately for
  `cuda:0` and `cuda:1`, with transfer excluded and clearly declared;
- attested physical identity for each indexed WX9100, including model, PCI BDF,
  PCI vendor/device ID, HIP UUID, and the pinned inventory SHA-256;
- profiler source identity as Git commit SHA plus `working_tree_clean: true`,
  independently recorded from checkpoint training source snapshot/digest;
- exact run configuration and runtime/toolchain versions.

Store the report next to the capture and checkpoint bundle and include its
digest in the project evidence/provenance ledger. Never overwrite a prior
report for a different checkpoint or capture set.

## Dependencies and ordering

This tranche can be implemented and contract-reviewed without GPU/Hailo access.
Its runtime profile specifically requires both WX9100s and an exclusive lease,
so the measurements must wait until current training releases the pair. Runtime
evidence also depends on (1) a completed five-track checkpoint bundle from the
intended source/protocol, (2) one fresh production request capture per track
produced through the actual adapter with capture explicitly enabled, and (3) a
later deliberate profile run with checkpoint and capture hashes rechecked.
Existing uninterrupted r4 runs use an older source and have no current
evidence that matching captures exist; do not treat their checkpoints as
compatible or change/stop that run to produce these artifacts.

The scorer profile planned here measures only captured-input model forward and
argmax on the two WX9100s. The next physical deployment gates remain separate:
Hailo operator compilation, device memory fit, and on-device latency on the
real target; measured host-to-device transfer; production request-age under
load; full AD* and candidate
construction, feasibility, arbitration, simulator replay, and robot
score-to-actuation timing. Phase 4 generalization and model qualification also
remain outstanding. No profiler schema or green synthetic check can satisfy
those evidence requirements.

## Acceptance criteria

- `--production-capture-dir` selects strict capture-backed scorer profiling;
  absent option preserves existing synthetic diagnostics.
- Capture mode requires the exact explicit `cuda:0`/`cuda:1` pair, an exclusive
  lease, and successful physical attestation of both pinned WX9100 identities;
  no CPU, one-card, P600, or generic-visible-device fallback is allowed.
- A valid run binds all five exact captures to their selected checkpoints,
  validates them before profiling, records consumed input digests and runtime
  provenance, independently records clean profiler commit identity and
  checkpoint training-source identity, and labels timed stages accurately.
- Any incomplete/malformed/mutated/mismatched capture set or any checkpoint
  identity mismatch fails the report and exits nonzero; no synthetic fallback.
- Documentation and ledger classify results as scorer-only evidence and keep
  Hailo/deployment qualification false until the corresponding target hardware
  and end-to-end gates are run.
