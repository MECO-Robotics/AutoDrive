# Phase 5 WX9100 queue balancing

**Scope:** queue-only device balancing for never-started MECO lanes in a
positively identified homogeneous WX9100/ROCm pool. This is a scoped summary,
not whole-plan approval or runtime qualification.

The planner approved a narrow supervisor improvement: when a requested
training device is occupied, a never-started lane with no checkpoint may use
an otherwise idle device from the explicitly requested equivalent WX9100/ROCm
pool. Existing checkpointed lanes remain on their checkpoint device. The
supervisor records each lane's affinity and selected device, reserves an active
evaluation device, and persists guarded per-lane config and run ID provenance.
Restart recovery validates the requested pool, source, protocol, configuration,
and any existing run manifest; conflicting or missing manifests fail closed.
Queued-manifest writes precede the pre-dispatch binding, and the dispatched
tombstone precedes process launch.

Implementation was completed on isolated branch
`codex/meco-wx9100-queue-balance` in commits
`bb5afa4b0c6fea4580ae58dec7fb3d657fd2cf58`,
`a13029762c4b97347436493b5255056236eb086b`,
`c3ecf482486432c90f1ce99f73e17215d8e464f5`, and
`ca27bc443c93f5a2656789b39ab954910a853e69`. The final implementer commit
`ca27bc443c93f5a2656789b39ab954910a853e69` received scoped reviewer approval
for future integration. No integration commit exists; the implementation is
not in live `main`.

Static compilation and `git diff --check` passed. No tests, runtime throughput
measurement, training, simulator evaluation, Hailo compiler/device check, or
hardware/deployment qualification was performed. The changed Python source is
part of the current checkpoint source identity, so existing checkpoints from
the previous source revision cannot resume under the strict source guard until
source continuity is explicitly resolved. This tranche establishes neither
throughput improvement nor training or evaluation results.
