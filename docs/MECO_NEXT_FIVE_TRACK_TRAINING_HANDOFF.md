# Next MECO five-track campaign handoff

Use this checkout as the source for the next matched five-track campaign after
the active run is deliberately stopped. The integrated source includes the
reviewed A1/A2/A3 action/log-probability fix, Plan A P-map propagation into
focal-robot AD* replanning, Phase 2 Plan A v3 warm-start support, Phase 4
factors, and Phase 5 scorer-timing correction. The current active workers
were not changed or restarted during this integration.

## Campaign setup

An offline manifest API now describes the five-track configuration and
validates source, protocol, device inventory, and isolated output paths. It
does not launch a campaign: `launch_ready` stays false until a supervisor
consumes the manifest and acquires shared device leases. Phase 2 coverage also
stays false until bound dataset/replay evidence is validated. See
`frc_defense.meco_training_manifest` and provenance tranche #44.

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
