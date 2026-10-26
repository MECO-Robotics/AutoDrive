# P5 PPO/RNG, typed end causes and atomic trainer package plan

**Date:** 2026-10-26  
**Status:** G0/source-authoring gate amendment candidate; previous plan head requires fresh independent exact-head review
**Planner:** plan_a_owner (planner only; not implementer, reviewer, or human approver)  
**Implementation source base:** 6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22  
**Original accepted overall plan:** 6325920252773cfa05d63167ba3c29a90debdb01; file SHA-256 9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a
**Accepted overall G0/source-authoring plan:** commit `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; exact Pass 1 ACCEPT record commit `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`.
**P0:** Accepted G0/source-authoring package plan `72ee8fce1d4a955d100b97d47b58528374faddec`, file SHA-256 `16c1fbcd881bd77adf178c8d8bf5f17bd3cf3e988cde767fbbc7731c8ff088d9`; exact Pass 1 ACCEPT record commit `cb02ef8fd74ab4b65c4b0f86a1796eaad616e77d`, record SHA-256 `a8c5164730d62bd04c7a134e0d395a81ce8f76c1088db6e5f5a8e2b5bd62701d`. This supersedes amendment REVISE `c4f136b267f8e41e0baec53394f28fac28f16eb5`. P0a decisions/candidate ACCEPT remain prerequisites.
**P1:** Accepted updated package plan `606ba5adce04afd84bfb372260aebde8fd4fcec6`, file SHA-256 `7dc5be35d3826a6047653021e97a941d6872a73a6997ba768e95753f693b0653`, exact Pass 1 ACCEPT record `77816edbd1923c018d4af774b2f1b3bae0f415b8`. P1 implementation/API candidate remains a separate pending SOURCE/API gate.
**P2:** Accepted G0/source-authoring package plan `fa15258776030ccb2ca5f9f637a38594075797d7`, file SHA-256 `123ff2f8741636eabc1aea4b876a59ac326fa81f4739a469ecc4271f4d6807ca`; exact Pass 1 ACCEPT record commit `cb02ef8fd74ab4b65c4b0f86a1796eaad616e77d`, record SHA-256 `10c1ac00586f9535b53905395afe34ca2fefe3cb062656ec38d2236a83db185c`. This supersedes amendment REVISE `09cf081a4b47b66a9b3fcecf3c3748eda5722465`. P2 implementation/API candidate remains a separate pending SOURCE/API gate.
**P3 Plan A:** Accepted G0/source-authoring package plan `2b06304bde63cace9f59b6b25497c82250cf759e`, file SHA-256 `92a57003e63b0aad90c95f45b23c591106d03071675ff5f58ec39a8367777e60`; exact Pass 1 ACCEPT record commit `9790c4ad2`, record SHA-256 `7f0ae276d1c48ffcd7a320440f8ff73d6b3022c01c2edc40faf0d23f20910b66`. This supersedes the REVISE records for prior heads `c2094a60484efa58e7e4b0a626398dcae71e665b` and `7f88e7c975ac264073b089c94ef2f2795aaeb9dd`. P3 implementation/API candidate remains pending.
**P4 Plan C:** Accepted updated package plan `c277571f1e286468445b912f8ce377279f4f4056`, file SHA-256 `b93e87b1fe1f18ef6aea99619f319cf0d91ffe0fa3c1d6747826f8e7cd754a0c`, exact Pass 1 ACCEPT record `1ca6c8f4e33f82a3c89ec76c2897336006a9f751`. P4 implementation/API candidate remains a separate pending SOURCE/API gate.
**P6 supervisor and qualification:** Accepted package plan `ea51dd8c84ab4e59bf056fae8c47facaee0c1fbd`, file SHA-256 `60cb8e6126cea8d307a96caef6b3099e5db5e43efe41170df0cb4e3237462a5e`, exact Pass 1 ACCEPT record `f97a9ed025b892e75669b95eec4785d15f05afff`, record SHA-256 `8bee77e4ce40dbc3be0058f61f08f992f539c8ed2a57945802263ae0eecc9710`. P6 implementation/API candidate remains pending.
This dependency ledger distinguishes accepted package plans from pending/revised gate amendments and unaccepted implementation APIs; it does not change P5 scope or unblock implementation.
**Accepted reconciliation:** 03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9; artifact SHA-256 294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c  
**Canonical profile SHA-256:** 5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3  
**V4 proposal SHA-256:** 0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6  
**Owner-adoption matrix SHA-256:** 9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53

## Objective and authority

Implement the accepted P5 transition/PPO caller integration for A1/A2/A3/C1/C2: the 160-physics-tick strategic horizon, at most 12 ticks per macro with a partial final macro, gamma 0.993 per macro under the adopted branch/continuation exponent convention, distinct terminal/truncation/bootstrap semantics, typed end cause, action/log-probability/command linkage, two independent GPU Philox streams, all-or-none PPO updates, and five source-bound checkpoint manifests and continuations.

P5 consumes the exact P1 standalone V4 transition/event/artifact API. It does not reimplement or alter P1 field order, profile, byte totals, event schema, reader, digests or manifest encoding. Overall plan 8ef and the P0, P1, P2, P3, P4 and P6 package plans are accepted at the exact heads listed above. Those plan ACCEPTs do not accept implementation/API heads: P0a still requires its authorized human decisions and candidate ACCEPT, and P1–P6 implementation interfaces remain separate SOURCE/API gates. Before P5 source authoring, this plan and P0a must be accepted, P1/P2/P3/P4 candidate interfaces must receive independent SOURCE/API ACCEPT, and exact heads must be pinned. Pre-G0 authoring cannot integrate or runtime-use any candidate API.

This is a package plan only. It authorizes no action until this exact amendment receives independent Pass 1 ACCEPT and its source prerequisites are pinned. Then isolated ZFS worktrees may author only allowlisted source/test files and perform static source/diff review before G0. A pre-G0 reviewer may return SOURCE/API ACCEPT or REVISE for an exact candidate head; this does not constitute package-final Pass 3 ACCEPT or permit dependent-package integration/runtime use. Do not touch live diagnostic campaign PID 2750253, its dashboard, workers, outputs, checkpoints, devices or leases. Until the user manually stops the campaign and both WX9100 leases are independently confirmed released, builds/compilation, test execution, simulator/runtime, numerical work, GPU/device use, dashboard, training, evaluation, checkpoint mutation, integration, merge, promotion and qualification remain prohibited. G0, P0a and all package-specific gates remain in force. Pre-G0 source work may cite pinned reconciliation/profile hashes but must not add or require the absent reconciliation artifact in its worktree; the reviewed contract-anchor bridge is post-G0 and required before validation/integration.

## Contract and source evidence

The adopted profile fixes one track per run (N=1), S=6, R=32, T=12, E=504, K=16, the V4 field table, a 314,469,051-byte payload and 96-byte envelope. P5 may not add, remove, reorder, resize or reinterpret fields. PPO stream/draw registries are source-bound identities and do not expand the row or payload byte count. P5 implements the reconciled PPO/RNG owner row, including its exact Philox schedule and digest coverage.

Read-only inspection at source base 6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22 found:

- frc_defense/meco_track_training.py:82-105 has configurable rollout steps, action_ticks=12, gamma=.993, and a CPU device default. train_meco seeds global Torch/Python RNG at lines 871-877 and optionally configures one CUDA/HIP device.
- _run_macro_step receives separate done and truncated tensors at lines 609-617 but converts them to Python booleans and stops on either at lines 666-670. The rollout stores one done=ended or timeout at lines 1201-1206.
- _gae at lines 696-705 uses that single dones mask for both TD bootstrap and recurrence. _ppo_update selects zero bootstrap whenever the last row is done at lines 779-783; its caller branches on rows[-1]["done"] at lines 1251-1278. This loses truncation bootstrap and end cause.
- _prepare_plan_a_decision and _prepare_decision sample with sample_candidate_action at lines 423-430 and 526-528. frc_defense/meco_tracks.py:580-586 uses implicit torch.distributions.Categorical.sample(). The trainer extracts sampled action, command, route target and heading via .item()/.tolist() at lines 1117-1138. It stores action/logprob separately from semantic command and per-tick execution trace at lines 1172-1206.
- _ppo_update validates replay/digests at lines 733-783, converts GPU epoch permutation to a Python list at lines 793-795 and calls optimizer.step per row at lines 806-828. A later failure can occur after earlier rows have already mutated parameters/optimizer state.
- Checkpoint helpers _atomic_json and _atomic_torch atomically replace individual files at lines 147-171. Resume loads schema v1 with map_location="cpu" at lines 920-950 and restores global Torch/Python and optional device RNG at lines 951-959. The update writer persists model, value head, optimizer and legacy RNG fields at lines 1309-1325, but not V4 Philox continuation or five V4 manifests.
- frc_defense/meco_counterfactual.py:415-427 calls _run_macro_step, moves reward to CPU, and returns only ended/timeout booleans rather than a typed end cause. Its state/transition digest and route/path writers include .numpy(), .cpu(), .item() and .tolist() paths, e.g. lines 150-160, 380-436, 464-467 and 534-567.
- The accepted P1 plan expressly excludes meco_track_training.py and trainer integration tests, provides a standalone typed V4 batch/manifest API, and assigns caller integration plus optimizer/RNG/checkpoint no-mutation tests to P5. Its exact API head must be independently accepted before P5 consumes it.
- The accepted PPO/RNG owner response at commit 602565ef99986996d2c12d06024833247dc3d395 (SHA-256 219f3466741c17d9e49a4d402f0b3f8ae6f016f0738a068034a8ad312f610d75) adopts two independent Philox4x32-10 streams and exact draw semantics. These decisions are the technical contract, not implementation evidence.

These citations establish current gaps only. No tests were run and no trainer, device, simulator, GPU, dashboard or campaign was inspected at runtime.

## Dependencies and worktree

P5 source-only authoring requires:

1. Independent Pass 1 ACCEPT of this exact amended plan and accepted P0a candidate plus required human decisions.
2. Exact P1 transition/event/artifact API candidate head with independent SOURCE/API ACCEPT; P1's accepted package plan alone is not an implemented API.
3. Exact P2 observation/route, P3 Plan A, and P4 Plan C candidate interface heads with independent SOURCE/API ACCEPT, plus pinned field/action order, observation/route inputs, output ownership and shared-file boundaries. Their package plans are accepted at P2 `fa15258776030ccb2ca5f9f637a38594075797d7`, P3 `2b06304bde63cace9f59b6b25497c82250cf759e`, and P4 `c277571f1e286468445b912f8ce377279f4f4056`; the P2/P3 amendments supersede prior REVISE heads. The implementation/API candidates are still pending for all three packages.
4. Distinct Pass 2 implementer, pre-G0 source/API reviewer and post-G0 final reviewer; all work occurs in an isolated ZFS worktree from the approved source base and candidate API references.

G0 is required before any build/compile, test execution, simulator/runtime, numerical work, calibration/evaluation, dashboard, device/GPU use, training, checkpoint mutation, integration, merge, promotion or qualification. It requires manual campaign stop, independent release confirmation for both WX9100 leases, explicit maintainer authorization, and a reviewed contract-anchor bridge before validation/integration. P0b is not needed to author P5 source; it is required before fresh training or qualification.

Create a fresh ZFS-backed worktree from approved source base 6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22, referencing exact pinned candidate interfaces but not copying/depending on the absent reconciliation artifact; proposed path .worktrees/autodrive-v4-p5-ppo-rng and branch codex/v4-p5-ppo-rng-YYYYMMDD. Never use, inspect, or branch from the live campaign checkout. Before G0, author only allowlisted files and static reviews. Candidate APIs may guide source authoring, but P5 and dependent packages may not integrate or runtime-use them until G0 and final package ACCEPT. P5 is sole owner of frc_defense/meco_track_training.py and owns its counterfactual end-cause integration in frc_defense/meco_counterfactual.py. Other packages may stage source against a pinned candidate API but cannot integrate/use it until post-G0 final ACCEPT.

## Finite Pass 2 allowlist

Pass 2 may edit only:

- frc_defense/meco_track_training.py — sole owner for rollout/transition assembly, terminal/truncation/bootstrap, fixed horizon, PPO validation/update transaction, sampled action and old-logprob binding, execution trace, update RNG counters, checkpoint/manifest caller integration, resume and device enforcement.
- frc_defense/meco_counterfactual.py — typed end-cause and fixed-horizon transcript/branch integration with GPU-resident state; no relabeling legacy data. P5-only edits here; model packages consume reviewed datasets and do not overlap source edits.
- frc_defense/meco_ppo_rng.py — new GPU Philox4x32-10 action/permutation streams, registry, key derivation, counter operations, categorical transform and Fisher–Yates schedule. Keep registry namespaces separate from gamepiece/physics RNG.
- frc_defense/meco_ppo_transaction.py — new staging/rollback helper for the complete model+optimizer+PPO RNG update, only if the trainer cannot provide a clear transaction without duplicating state. If unnecessary, do not create it.
- frc_defense/csrc/meco_ppo_rng_hip.cpp and frc_defense/csrc/meco_ppo_rng_hip_kernel.hip — new HIP implementation/registration only if profiling or backend support requires a custom Philox kernel. If Torch HIP ops suffice, do not create them. No build/configuration files are authorized; dynamic registration must use the listed Python module. Stop for plan amendment if that cannot be done safely.
- tests/test_meco_ppo_rng_v4.py — new known-answer, registry, schedule, permutation/rejection, overflow/collision and exact replay fixtures.
- tests/test_meco_training_v4.py — new transition validity, action/logprob/command linkage, terminal/truncation/bootstrap/GAE, horizon and all-or-none update fixtures.
- tests/test_meco_counterfactual_v4.py — new typed cause, partial final macro, replay identity and fixed-horizon fixtures.
- tests/test_meco_checkpoint_v4.py — new five-manifest, source/profile/protocol/device binding and exact resume/failure fixtures.
- tests/test_meco_training_v4_simulator.py — new seeded end-to-end evaluation exercising P5 transition/end-cause/horizon/action trace and score integration, with fresh replay inspection.

No other source, test, schema, build, dashboard, data, checkpoint, output or campaign path is authorized. In particular P5 does not edit P1 transition/event/manifest encoder files or score producers, P2 observation/route files, P3/P4 model/support files, frc_defense/meco_tracks.py, P6 supervisor/qualification code or any unlisted path. P5 calls its own RNG API from the trainer path; it does not modify the shared sampling helper in meco_tracks.py. If a required change lies outside the allowlist or collides with a pending package owner, stop and request an exact-head Pass 1 amendment.

## Required technical work

### Fixed horizon, typed cause, validity and GAE

- Enforce exactly 160 physics ticks. Each macro executes at most 12 ticks; the final macro may execute the remaining 1–11. Never overrun. Preserve frozen episode seed and generation identity at reset/branch boundaries.
- Apply gamma 0.993 per macro under the adopted convention: first branch action exponent 0; each continuation decision adds one exponent; physics ticks do not increment it individually. Keep gross strategic reward separate from route/risk costs. Golden fixtures cover the first macro, continuation, terminal and partial final macro.
- Carry terminal and time-limit truncation separately even when both are true; terminal takes precedence for macro status and bootstrap. Carry transition_finished, valid-bootstrap, reset and typed end-cause status through step, counterfactual transcript, rollout, replay and P1 transition output. Use the exact accepted registry; do not invent status meanings or change P1 fields.
- Preserve row_valid, actor_valid, critic_valid, command_valid, terminal, truncated, transition_finished, bootstrap_continuation and reset masks. Padded rows affect no GAE, replay, RNG, optimizer or diagnostics. GAE may use the value of a valid final observation for a truncation delta, but recurrence stops across terminal, truncation or reset. True terminal and execution failure have no bootstrap. Execution failure may train critic only when observation/reward/value are valid. Invalid/nonfinite required values fail the whole transition/update.
- A nonterminal rollout boundary requires a valid next value. Truncation bootstraps only from a valid final next observation. If both end bits are true, terminal wins. Reset/sequence/generation boundaries prevent advantage leakage to a new episode.

### Action and command linkage

For each valid stochastic decision, preserve the exact family-local policy index/source row and old log probability under the same frozen logits/support/sample. Preserve semantic action_id separately from the local index and the fixed 12-tick executed trace. Remasking, fallback, preemption, target loss, zero executed ticks or a late event never replaces the sampled action or recomputes its old log probability. Such failure invalidates actor credit; critic use requires valid inputs. Bind row/command identity to accepted P1/P2/P3/P4 APIs. Deterministic bootstrap/evaluation consumes no PPO random words.

### GPU Philox contract

Implement the adopted PPO RNG row without global PyTorch/Python RNG or CPU fallback:

- Stream registry: uint16 codes 0 INVALID, 1 ACTION_CATEGORICAL, 2 EPOCH_PERMUTATION. Separate draw-site registry: 0 INVALID, 1 ACTION_CATEGORICAL_WORD, 2 PERMUTATION_FISHER_YATES_WORD. These are source identity and preserve V4 bytes. They are separate namespaces from rng_protocol_code=2 gamepiece/physics Philox.
- Derive each 64-bit key by SHA-256 over ASCII domain meco-ppo-v4-rng-v1, root seed as uint64 little-endian, track code and stream code; first eight digest bytes form the little-endian key. Each stream has an independent 128-bit word counter; serialize words little-endian, count rejected words, and fail on collision or wrap. Fresh counters start at zero; episode/chunk reset does not reseed.
- Action stream consumes one word per valid stochastic decision in rollout-row order before execution outcome. Use canonical support order; convert finite float32 logits to float64, subtract maximum, exponentiate and normalize with a fixed ascending-index sum. Set u=(word+0.5)/2^32; choose the first legal category whose cumulative probability is strictly greater than u; final legal category absorbs rounding remainder. Invalid/padded/no-support rows consume no draw and fail per V4 validity rules.
- Epoch stream uses descending Fisher–Yates over valid rows ordered by epoch then rollout slot. For i=n-1 down to 1, consume uint32 words until word < floor(2^32/(i+1))*(i+1), set j=word mod (i+1) and swap. Rejected words advance the counter. Valid critic-only rows participate; padded rows do not. No other P5 draw sites exist in this version.
- Bind registries, code mappings, transform math, invocation order and source identity. Resume restores counters before stochastic draws and rejects wrong source/profile/backend/device, missing/malformed counters, collision, overflow or unknown codes. It never silently reseeds. Bit-exact continuation is required only for the same pinned source/profile/backend/device and checkpoint.

### Atomic PPO update and five manifests

- Consume P1's standalone V4 transition/artifact API. Validate schema, capacity, source/config/profile/protocol/device identity, rows, action/logprob replay, digests, events and RNG before optimizer mutation.
- Stage the update or retain complete rollback state so any validation, digest, capacity, schema, replay, RNG, loss, gradient, optimizer or checkpoint failure leaves model/value-head parameters, optimizer slots/steps, both PPO stream counters and published latest checkpoint at pre-update state. No earlier row/epoch may remain committed after later failure.
- One update is the atomic model+value-head+optimizer+two-stream RNG+transition+checkpoint boundary. Publish task/stat/report references only after it succeeds. Keep last good checkpoint until complete new checkpoint and P1 manifest are durably staged and atomically published. No manifest may point to partial/missing bytes.
- A simulator tick cannot be undone. If failure follows a tick, stop the worker, do not credit a partial transition, preserve last committed checkpoint and reconstruct the environment from frozen task seed on resume. Do not claim simulator rollback or continue with uncommitted model/RNG state.
- Produce exactly five manifests, one for each A1/A2/A3/C1/C2. Bind checkpoint bytes, policy/value model, optimizer, root seed/stream codes/counters, source/config/profile/protocol, track and device identity/runtime. The bundle has five unique compatible manifests. Resume rejects missing, duplicate, stale, malformed or mismatched manifests and legacy schema-v1 checkpoints. P5 consumes P1's accepted writer/schema; it does not create a parallel manifest format.
- Do not change P1 V4 row/schema/manifest bytes. If the accepted API cannot bind P5 continuation state, pause for a separately reviewed V4 replacement; do not silently extend bytes.

## GPU-only execution boundary

All algorithmic tensors and work in trainer, optimizer, policy sampling, GAE, returns, replay validation, digest inputs, counterfactual reconstruction, state checks and RNG must run on assigned HIP/WX9100 devices after G0 validation begins. No CPU simulation, policy inference, scoring/reward, route/state digest, GAE/metrics, dynamic .item()/.tolist()/.cpu()/.numpy() reads, host iteration over device rows, global RNG fallback, P600 fallback or silent migration is permitted. Host Python may supervise processes, parse immutable configuration, send control messages and persist opaque bytes; it may not inspect dynamic tensor values or compute algorithmic results. Before G0 none of this numerical/runtime work may execute.

Pin each worker to an explicitly released WX9100 UUID and HIP index, recording ROCm/HIP/PyTorch/runtime, process, track, kernel and device identity. Across the five lanes, use both WX9100s without oversubscribing into CPU work. Device loss, wrong UUID, unsupported kernel or ambiguous affinity stops the worker. Do not claim one simulation jointly uses both devices unless the integration implements it. Evaluation shares and checkpoint isolation follow the accepted overall plan.

Preserve the user's confirmed current counterfactual collection defaults at source base 6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22: horizon_ticks=160, action_ticks=12, candidate_k=16, prediction_mode="cv", max_c_candidates=8, max_a_pairs=8, gamma_per_macro=0.993 (frc_defense/meco_counterfactual.py:920-924; DEFAULT_GAMMA at line 46). The callable's default device="cpu" is not an allowed execution path; any future runtime must explicitly bind a supported WX9100 and fail closed on CPU/P600 or ambiguous affinity. Changing these algorithmic defaults requires a separately reviewed plan/source decision.

## Acceptance evidence

All validation is future-gated; this plan ran none.

- Deterministic GAE fixtures cover padding, valid/actor-invalid/critic-only, terminal, truncation, both end bits, continuation, reset, invalid next observation and execution failure. Verify precedence, bootstrap delta and no cross-boundary recurrence.
- Horizon fixtures cover exactly 160 ticks, thirteen full macros plus a 4-tick final macro, no overrun, exponent 0 for first branch and one increment per continuation, separate route/risk costs, early terminal and time-limit truncation.
- Action/replay fixtures preserve local index/source row, old logprob and command identity while execution trace/remask/preemption stays separate; critic-only rows get no actor credit; deterministic paths consume no random words.
- Philox fixtures cover standard Philox4x32-10 known-answer vectors, domain/key bytes, registries, categorical boundaries, Fisher–Yates accepted/rejected words, epoch/row order, reset/resume counters, collision/wrap failure and no counter change after injected errors.
- Atomicity fixtures inject failures at prevalidation, digest, capacity/schema, replay, each epoch/row, gradient/optimizer, manifest staging, serialization and final replace. Assert model/value-head/optimizer/RNG and checkpoint are unchanged after rejected updates. Validate exactly five manifests and reject wrong/missing/duplicate/legacy identities.
- Counterfactual fixtures preserve typed end cause and replay identity, horizon/partial final macro and reject invalid cause/hash/sequence. Do not relabel old outputs.
- Tests exercising robot, scoring, episode ending or game behavior pair with a seeded end-to-end simulator run. Inspect fresh replay and record horizon, cause, sampled-vs-executed trace and score linkage; report scenario, seed, replay ID and outcome.
- Before any dashboard replay, restart services from current accepted integration checkout and verify backend/frontend asset identity, active services and successful endpoints. Generate a fresh replay, inspect provenance and mark old-code replays outdated.
- Run approved P5 tests and simulator cases on both assigned WX9100s; retain UUID/runtime, per-kernel placement/transfer/synchronization, Philox and optimizer device evidence, with proof of no CPU/P600 fallback. Report tests, simulator scenario/seed/replay, dashboard checks, device traces and gaps.

Review is staged. Before G0, an independent source reviewer checks exact commit/blob hashes, allowlist, P0a/P1/P2/P3/P4 candidate bindings, V4 field/byte invariants, Philox source contract, atomicity design and manifest interfaces and may return SOURCE/API ACCEPT or REVISE. SOURCE/API ACCEPT pins a candidate only; it is not package-final Pass 3 ACCEPT and does not authorize integration or runtime use by P5 or dependent packages. After G0, final Pass 3 checks the same exact head plus executed tests, failure injection, five manifests, both-device traces, seeded simulator replay and current-dashboard evidence. Only final exact-head ACCEPT permits integration; a rebase or source change requires renewed review. P5 final ACCEPT still does not authorize training: P0b, fresh source-bound checkpoints and separate campaign authorization are required.

## Risks and stop conditions

- P0/P1/P2/P3/P4/P6 package plans are accepted at the exact ledger pins; implementation/API heads remain separate pending gates. Source-only work requires exact candidate-interface SOURCE/API ACCEPT; execution and integration remain blocked until G0 and final Pass 3 ACCEPT.
- End-cause status must use accepted V4/P1 registries; no local enum renumbering, extra fields or byte changes.
- Rollback of model/Adam state can require substantial memory. Measure it, but do not weaken atomic semantics to reduce memory.
- PPO rollback cannot undo simulator ticks. Stop and reconstruct only from the last committed checkpoint and frozen task seed.
- HIP integer math and float transforms can vary by backend. Pin source/profile/backend/device and known-answer fixtures; reject incompatible continuation instead of using framework RNG.
- Existing code has many scalar/list/CPU digest paths. Audit each call graph; only process control, immutable config and opaque persistence may remain on host.
- P5 may not change P1 schema/manifests. If continuation state cannot fit the accepted API, stop for reviewed replacement.
- This plan does not satisfy G0 or authorize any action against the live campaign.

## Three-pass roles and disposition

- Pass 1 planner: plan_a_owner; cannot implement or review P5.
- Pass 2 implementer: unassigned; must use exact accepted dependency head and finite allowlist.
- Pass 3 reviewer: unassigned, distinct from planner and implementer; reviews exact pushed heads and persists ACCEPT/REVISE.

Planner disposition: **PENDING independent exact-head Pass 1 review of this gate amendment.** No source implementation, executed tests, device use, simulator/dashboard validation, training, checkpoint qualification, human approval or Stage 1 result is claimed.
