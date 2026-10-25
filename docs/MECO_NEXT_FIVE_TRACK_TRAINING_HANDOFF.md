# Next MECO five-track campaign handoff

An offline r5 config and manifest are prepared at
`/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDriveCampaigns/manifests/`.
The earlier r5 manifests bound to `b02db09f3`, `9830c2647`, and `85618385` are
stale after the reviewed Plan A route and Plan C corpus-audit source updates.
The current files are
`meco-five-continuous-20261025-r5-plan-c-audit-5389.config.json` and
`meco-five-continuous-20261025-r5-plan-c-audit-5389.json`, generated and
validated against clean source commit
`5389e618a5aad99b32753de20fb9c1d47cdce888` in worktree
`/mnt/zfs-storage/All Drives/Brian Data/Projects/.worktrees/autodrive-meco-r5-current-source-5389`.
The canonical manifest digest is
`87c5abcb30cacaa6b4cd612d31028a9c1ab5215950a12be46a57b7c40c73c542`.
Do not launch a new campaign while r4 owns the two WX9100 leases. The integrated source includes the
reviewed A1/A2/A3 action/log-probability fix, Plan A P-map propagation into
focal-robot AD* replanning, Phase 2 Plan A v3 warm-start support, Phase 4
factors, and Phase 5 scorer-timing correction. The current active workers
were not changed or restarted during this integration.

**Manifest refresh required:** the r5 config and manifest above predate the
Phase 2 complete-set coordinator integrated through `59524d21a`. They are now
stale for launch and must be regenerated from a clean current checkout after
the active r4 run releases both device leases. The coordinator and supervisor
now preserve and forward an exact all-five warm-start map, but no dataset,
warm-start weights, or complete-set seal has yet been generated. Keep
`phase2_ready`, validation readiness, and qualification readiness false until
the runtime campaign and later gates produce their required evidence.

## Campaign setup

An offline manifest API now describes the five-track configuration and
validates source, protocol, device inventory, and isolated output paths. The
supervisor now consumes that manifest and acquires shared device leases with
the validation runner. Launch requires explicit `--diagnostic-launch`.
Fresh manifest roots are claimed only after device leases; explicit
`--resume-campaign` now resumes only when campaign roots, runtime/config/source
bindings, and durable evaluation state validate. Phase 2 coverage stays false
until bound dataset/replay evidence is validated.
See `frc_defense.meco_training_manifest`, `frc_defense.meco_supervisor`, and
provenance tranches #44 and #46.

The current replacement manifest enables Phase 4 training exposure and freezes
all 466 protocol validation cases, each with its exact 100-seed list. Its
canonical manifest digest is recorded above. Offline manifest validation
passed against the pinned source worktree. Both campaign output roots are absent.
Launch readiness, Phase 2 coverage, validation readiness, qualification
readiness, and warm starts remain false.
The reviewed durable queue is now integrated at `53157b4f8`: it preserves the
ordered schedule, five-checkpoint snapshots, and exact case across failure or
restart; it verifies report/status/history before advancing and applies a
persisted failure-time retry delay. The no-loss dispatcher blocker is resolved
in code, but the full 466-case schedule has not run. Hold launch until r4
releases both device leases. The replacement manifest is pinned to current
source and must be revalidated at launch time.

The current runtime also does not meet the user's CPU-free request. The two
WX9100s run the model and much tensor state, but Plan A graph routing moves
work into Python/CPU; orchestration and output serialization are CPU work by
design. Fuel contact coloring also has a Python fallback when `solver="colored"`
is selected, while the current default is `jacobi`. There is no manifest
setting that makes the runtime CPU-free. Any future GPU-hot-path rewrite needs
dedicated implementation and seeded simulator/replay evidence.

- Create one frozen supervisor configuration for A1, A2, A3, C1, and C2; bind
  the training protocol, qualification protocol, source identity, runtime and
  device, seed schedule, update frontier, and per-track configuration hashes.
- Start all five tracks from fresh initialization under the same source
  identity. Do not resume checkpoints from the previous source: shared Python
  source changed, so exact-resume validation should reject those artifacts.
- If enabling supervised warm start, use only artifacts collected and fitted
  from this exact source identity and frozen training protocol. Plan A v3 is
  available for A1/A2/A3; it supervises sampled U+B returns, leaves P
  unlabelled, and does not provide exhaustive node coverage. Record each
  weights artifact hash in the supervisor configuration.
- The selected A1/A2/A3 P map now reaches the native focal slot-3 AD* planner
  on eligible replans. CUDA stream-captured calls with a P override raise and
  abort; no graph-compatible fallback is implemented. Require a seeded
  replay that inspects executed routes before treating P route behavior as
  validated. P oracle targets and exhaustive coverage remain unavailable.
- Keep the factor matrix separate from training. Phase 4 evaluation requires
  the documented supported scenarios/categories, identical checkpoint bundle,
  seeds, controls, and device across factor and no-factor conditions.
- Save the supervisor configuration, source hash, per-track checkpoint hashes,
  seed assignments, and dashboard ledger snapshot with the campaign outputs.

## Evidence status

The merged changes have feature-branch and integrated static review, but no
simulator evaluation has been run for this integration. Feature runtime
behavior, matched five-track outcomes, Phase 4 factor results, calibration,
and qualification remain unverified; do not treat the timing correction as
physical or hardware profiling evidence.
