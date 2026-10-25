# GPU-resident MECO score-event buffer tranche

**Pass:** 1 — planner scope  
**Date:** 2026-10-26  
**Plan base:** `origin/main` `d30cdd89b4d404d545254423e9d5bc950804e346`  
**Candidate dependency:** approved cumulative PPO/reset/pressure source `54bd1dc4651b039e2924e602c2ddf13a165536af` remains a separate integration dependency.  
**Status:** planning only; no source or runtime behavior is changed by this artifact.

## Decision

Implement a bounded GPU-resident buffer for accepted simulated FUEL score events so the existing replay-bound score ledger can be used with qualifying HIP physics. This closes the explicit fail-closed gap left by the approved gamepiece physics tranche. The scope is event telemetry and report materialization only: score decisions, score counts, rewards, RNG progression, physics, and event acceptance must remain identical.

This is the next narrow code gate because the current source explicitly rejects a score recorder in `_gpu_qualifying` mode, preventing qualification episodes from producing the plan's replay-reconstructable score outcome. It does not duplicate PPO, reset/cadence, pressure state, or gamepiece physics implementation. It extends the existing score-ledger capture seam with a device buffer and passive host drain.

## Evidence and current contract

- `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md` states at its current top entry that GPU score recording is rejected because event extraction materializes device values on the host; it names this as an outstanding blocker.
- `docs/implementation-provenance/gpu-gamepiece-fixed-mask-reset-rng-20261026.md` records exact Pass 3 approval of the GPU gamepiece physics tranche, but explicitly limits qualification when the CPU score-event recorder is enabled.
- `frc_defense/gamepiece_actions.py::_capture_fuel_score_awards` clones accepted masks and piece generations into a Python pending list. `_flush_fuel_score_awards` converts masks with `nonzero(...).tolist()` and reads tick, time, team, and generation through `.item()`; it deliberately raises for qualifying physics.
- `frc_defense/fuel_physics_runtime.py::advance_with_fuel` rejects the recorder before a qualifying step. `frc_defense/meco_track_training.py` also rejects recorder creation in qualifying mode and currently drains terminal counters to CPU at episode reporting.
- The pure contract in `frc_defense/meco_simulated_score_ledger.py` is already implemented: schema `meco-simulated-score-ledger-v1`, recorder cap `MAX_SCORE_EVENTS_PER_EPISODE = 32768`, accepted award fields (tick, elapsed match time, world, alliance, robot slot, piece slot, piece generation, static source site), ordered event digest, source identity and approval binding, replay reconstruction, and evaluation/replay cross-checks.
- Current model/source validation remains provisional. The buffer must not substitute for a human-approved estimand, official match score, seeded simulator evidence, or qualification.

## Narrow source scope

Expected implementation owners:

1. `frc_defense/gamepiece_actions.py`: write already-accepted score masks to fixed-shape device buffers, in deterministic event order, without Python pending tensors, host scalar reads, or host branches on device outcomes.
2. `frc_defense/_tensor_3v3_gamepieces.py` and the defense wrapper that invokes `_flush_fuel_score_awards`: preserve both roster call paths and define a passive reporting drain boundary.
3. `frc_defense/fuel_physics_runtime.py`: remove the recorder-only rejection after verifying the qualifying buffer contract; retain all unrelated backend/device/shape/graph guards.
4. `frc_defense/meco_track_training.py`: bind per-episode buffer lifecycle to the existing episode ID, seed, scenario, source identity, estimand approval digest, evaluation record, and replay export; materialize only after the episode's gameplay and physics decisions have completed.
5. `frc_defense/meco_simulated_score_ledger.py`: only if required to add a deterministic ordered-batch ingestion API that preserves the existing event validation/digest contract. Do not change score rules, included outcomes, schema semantics, or approval requirements.
6. Focused tests for buffer mechanics and unchanged ledger/replay binding.

Do not alter physics kernels, scorer acceptance masks, rewards, score counters, piece generation, RNG, PPO/checkpoint state, score rules, dashboard APIs, or active training manifests. If an unrelated qualifying-path guard must change, the implementer must stop and request a separate reviewed scope rather than widening this tranche.

## Required buffer contract

- Use fixed device tensors indexed by world and event slot, with a device-side per-world count and overflow flag. No device-derived dynamic allocation or host branch is permitted during a simulation decision.
- Capacity must be derived and justified per episode/world against the existing 32,768-event contract and the configured horizon/maximum score acceptance rate. If the bound cannot be proven for a supported configuration, retain the existing fail-closed qualification rejection for that configuration.
- Record only events whose existing accepted-score mask is true. Store all ledger fields losslessly: tick, match elapsed time, world slot, alliance, robot slot, piece slot, generation, and a deterministic source-site code mapped to the existing static source-site string on host.
- Preserve the current recorder's stable ordering: the same tick/update call order, then the existing world/slot/piece order for each accepted mask. Store enough sequence information to reconstruct that ordering exactly.
- A device-side capacity overflow must never truncate silently or affect gameplay. It sets a sticky per-world overflow status; terminal report construction then marks the score outcome incomplete and qualification-ineligible. Do not throw or read the status in the live step path.
- Event buffers are episode-scoped. Reset only the selected world's buffer at its accepted episode boundary; preserve other worlds' pending rows and counters. Bind rows to the same episode, seed, scenario, source identity, and approval digest used by the existing score context.
- Drain/materialization is a passive episode-report boundary after the last score/physics mutation for that episode. Only there may code copy the bounded event rows and overflow status to CPU and call the existing recorder/ledger contract. The drain must not decide whether an award occurs, reset physics, advance RNG, affect another world, or change a later action.
- Preserve current fail-closed behavior for unsupported shapes, devices, missing graph capture, missing approval, invalid source binding, and malformed report/replay data.
- Continue to emit `simulated_score_ledger`, `simulated_match_score`, `official_match_points: null`, and the same `score_estimand_approval_sha256` binding. Do not make a ledger itself qualification approval.

## Three-pass execution and acceptance

### Pass 2 — implementer

Implement only the contract above in an isolated worktree based on the approved/integrated source head. Keep active training checkout, supervisor, checkpoints, configuration, output roots, leases, and manifests untouched. Produce incremental commits and report exact final SHA and touched paths. Recompute source identity; do not claim that checkpoints from another source can resume unchanged.

### Offline acceptance (no simulator claims)

- Unit/source-contract coverage proves fixed capacity, per-world isolation, selected reset behavior, stable event ordering, exact field preservation, event digest parity with `MecoScoreEventRecorder`, source/approval binding, episode drain semantics, overflow fail-closed reporting, and no path from buffer status to score/reward/RNG/gameplay.
- Static call-path review covers both the 3v3 and defense wrappers, all score award sites, episode completion/censoring, replay bundle assembly, and evaluation-record binding.
- Exact-head independent review confirms there are no `.item()`, `.tolist()`, `.cpu()`, NumPy conversions, or host branches on device event masks/counts during the live qualifying physics/action path. The only device-to-host event transfer is the explicitly identified passive post-episode report drain.
- Existing non-qualifying recorder behavior and ledger schema/hashes remain compatible. A test row-for-row comparison demonstrates the same event order/digest and reconstructed score for equivalent fixed event fixtures.

### Pass 3 — reviewer

Independent review must verify the implementer head, accepted plan digest, tests/results, event ordering proof, capacity derivation, and explicit source/report lifecycle. Separate code review from runtime evidence.

### Runtime gates after implementation approval

Runtime validation is not authorized by this plan. Before any qualifying simulator execution:

1. Obtain the user's manual stop of live training, verify both WX9100 leases are released, and use an exclusive approved device lease. Never edit or restart the active training source to test this patch.
2. Resolve checkpoint continuity explicitly. The implementation changes source identity; do not resume old checkpoints under the new source unless an approved exact compatibility/continuity contract says so. Otherwise start a new source-pinned campaign after user direction.
3. On the exact reviewed checkout, run the focused tests, then a representative seeded simulator evaluation exercising multiple score events, multiple worlds, score/no-score/censored episodes, both wrappers, replay export, and ledger reconstruction. Inspect the fresh record and replay, retain seed/scenario/source/approval digests, and report result plus the exact test list.
4. Restart the dashboard on that checkout before replay review; verify service health, `/api/curriculum-status` provenance and `/api/meco-status` campaign endpoint separately, and backend/frontend asset hashes. Generate the replay only after currentness verification.
5. If score approval is absent or the approved estimand changes, do not collect or interpret qualifying outcomes. Keep official points null and qualification false.

## Risks and stop conditions

- **Capacity uncertainty:** the 32,768 bound may not cover every supported episode configuration. Require a mathematical/config-derived bound or keep unsupported configurations rejected.
- **Ordering drift:** vectorized writes can reorder simultaneous awards and change the event digest. Sequence by existing update/source call order and canonical indices, then prove exact parity.
- **Episode reuse:** vectorized training can auto-reset individual worlds while other worlds continue. Any ambiguity about the report boundary or buffer lifetime is a blocker; never drain/reset all worlds together.
- **False telemetry influence:** an overflow check or failed drain must not alter live score, reward, physics, or RNG. It only makes the passive evidence incomplete and qualification-ineligible.
- **Protocol/source drift:** changing ledger schema or source identity invalidates prior source-bound fixtures and approvals unless explicitly re-reviewed.
- Stop if the implementer must alter score rules or a game decision, if a buffer can overflow without a report-visible fail-closed state, if event order cannot be preserved, if episode-level drain cannot be separated from live decisions, or if device ownership/checkpoint continuity is unresolved.

This plan is a bounded source-code proposal only. It establishes no runtime behavior, score outcome, training continuation, replay, device parity, dashboard currentness, promotion, or qualification.

