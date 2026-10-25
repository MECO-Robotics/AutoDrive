# Next MECO five-track campaign handoff

An offline r5 manifest and config are prepared at
`/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDriveCampaigns/manifests/`
for source commit `b02db09f330642ad5498b6e76752d1df582c9089`. The clean source
worktree is `/mnt/zfs-storage/All Drives/Brian Data/Projects/.worktrees/autodrive-meco-r5-full-curriculum`.
Do not launch this campaign while r4 owns the two WX9100 leases. The integrated source includes the
reviewed A1/A2/A3 action/log-probability fix, Plan A P-map propagation into
focal-robot AD* replanning, Phase 2 Plan A v3 warm-start support, Phase 4
factors, and Phase 5 scorer-timing correction. The current active workers
were not changed or restarted during this integration.

## Campaign setup

An offline manifest API now describes the five-track configuration and
validates source, protocol, device inventory, and isolated output paths. The
supervisor now consumes that manifest and acquires shared device leases with
the validation runner. Launch requires explicit `--diagnostic-launch`, and
manifest mode claims fresh roots only; it cannot resume an existing campaign.
Phase 2 coverage stays false until bound dataset/replay evidence is validated.
See `frc_defense.meco_training_manifest`, `frc_defense.meco_supervisor`, and
provenance tranches #44 and #46.

The prepared config enables Phase 4 training exposure and freezes all 466
protocol validation cases, each with its full exact 100-seed list. Its
canonical manifest digest is
`cbc18e88e7d967cf3770bec3335fa6f8963a11e58d0b2d3364684f3ffe98cb1f` and offline
validation passed. Launch readiness, Phase 2 coverage, validation readiness,
qualification readiness, and warm starts remain false. The evaluation queue
currently replaces pending jobs when it falls behind and may advance past a
failed or incomplete evaluation. A reviewed durable no-loss dispatch/recovery
change is required before claiming all 466 scheduled cases will execute. An
independent reviewer confirmed the manifest's internal bindings but rejected
it as a full-coverage execution guarantee until this is fixed.

The current runtime also does not meet the user's CPU-free request. The two
WX9100s run the model and much tensor state, but Plan A graph routing and fuel
contact coloring still move work into Python/CPU; orchestration and output
serialization are CPU work by design. There is no manifest setting that
changes this. Any future GPU-hot-path rewrite needs dedicated implementation
and seeded simulator/replay evidence.

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
