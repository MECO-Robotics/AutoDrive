# Independent Pass 1 review: P5 PPO/RNG trainer package plan

- **Verdict:** ACCEPT
- **Review date:** 2026-10-26
- **Reviewer identity:** `plan_c_owner`
- **Reviewer task:** independent exact-head Pass 1 review assigned by `/root`
- **Plan author:** `plan_a_owner` (planner role)
- **Reviewed plan commit:** `2742920fc22a87865bdcd03f162ef3a3b9a67978`
- **Reviewed file:** `docs/plans/meco-v4-p5-ppo-rng-trainer-package-plan-20261026.md`
- **Git blob:** `7d64310b00b88358aa05f2a942421cda9b7a65bd`
- **Plan file SHA-256:** `b9dee65073c9118745cf0cc0ab08606ff42d5220cf6ec0e25701829d11d30e62`
- **Accepted overall plan:** `6325920252773cfa05d63167ba3c29a90debdb01`, file SHA-256 `9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a`
- **Accepted P0 package plan:** `13dbd7f8f953995850e811202f7e9c5264c49dca`, file SHA-256 `09ec722d64d01d99b60cc9ed52e47ce68790e548f66107cc1a19dd9a7a588adf`
- **Accepted P1 package plan:** `53686a254318484f9c6bc3496ecc1c461572c53a`, file SHA-256 `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076`; P1 API implementation head remains a separate prerequisite.
- **P2 package plan:** `8d0727f10d46bfc5fee2810a1ec7fa7987ecaf91`, pending independent Pass 1 review at plan authorship; accepted implementation interface is not yet available.
- **P3/P4 interfaces:** no accepted exact interface heads are pinned; plan correctly makes both prerequisites for P5 Pass 2.
- **Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted reconciliation:** `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`
- **Canonical profile SHA-256:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`
- **V4 proposal SHA-256:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`
- **PPO/RNG owner response:** `602565ef99986996d2c12d06024833247dc3d395`, file SHA-256 `219f3466741c17d9e49a4d402f0b3f8ae6f016f0738a068034a8ad312f610d75`, ADOPT in the accepted reconciliation.

## Review scope and evidence

The plan preserves the accepted V4 dimensions and byte totals and consumes the standalone P1 API without reimplementing its field order, schema, event table, byte layout, digest, reader, or manifest encoding. It assigns the shared `frc_defense/meco_track_training.py` exclusively to P5, matching the overall plan's P5 ownership. The finite allowlist assigns counterfactual end-cause integration, the PPO RNG implementation, optional transaction helper and optional HIP kernels, and scoped fixtures. It explicitly excludes P1 schema/manifest sources, P2 environment/route sources, P3/P4 model paths, the shared model helper, P6 qualification source, and all unlisted paths.

The P5 contract matches the accepted PPO/RNG row: 160 physics ticks, at most 12 ticks per macro with a 4-tick final macro, gamma 0.993 per macro with branch exponent zero and one increment per continuation decision; distinct terminal and truncation flags with terminal precedence; valid truncation bootstrap and no cross-reset GAE recurrence; sampled action/log-probability kept separate from semantic command and executed trace; and an all-or-none model, optimizer, RNG-counter and checkpoint update. The plan adopts two independent Philox4x32-10 streams and the accepted uint16 stream/site registries, key derivation, categorical transform, descending Fisher–Yates schedule, rejected-word accounting, overflow/collision behavior, and resume constraints. It requires exactly five source/profile/protocol/device-bound track manifests and fails closed on incompatible or missing continuation state.

Read-only inspection at the pinned source base supports the stated gaps: `_run_macro_step` returns separate `done`/`truncated` but stops on either (`meco_track_training.py:609-670`); `_gae` uses one mask for both bootstrap and recurrence (`696-705`); PPO zeroes the final bootstrap on `done` (`779-783`, caller `1251-1278`); rollout collapses end bits into `done` (`1201-1206`); action/command extraction includes host scalar/list reads (`1117-1144`); per-row optimizer steps occur before all epochs finish (`793-828`); and checkpoint schema v1 restores global RNG state and lacks the adopted V4 continuation (`920-959`, `1309-1325`). These are source observations, not claims of implementation or runtime validation.

The dependency gates are explicit and consistent with the top-level plan: P0a and G0 precede implementation; P0b is required before fresh training/qualification, not P5 source implementation; P1's exact API implementation must be independently accepted; and P2/P3/P4 accepted interfaces must be pinned before P5 Pass 2. The plan does not infer interface compatibility from pending plan names. P5 Pass 2 and Pass 3 roles remain unassigned and must be distinct from the planner and each other.

The no-CPU-compute requirement is explicit for trainer numerics, PPO/RNG, GAE, replay validation, digests, counterfactual reconstruction and metrics, with host access limited to process supervision, immutable configuration and opaque persistence. It specifies assigned WX9100 identity and disallows CPU/P600 fallback or silent migration. Acceptance requires deterministic transition/GAE, horizon, action linkage, Philox, atomicity, counterfactual and manifest fixtures; representative seeded simulator replay with outcome inspection; current dashboard/backend/frontend verification and fresh replay provenance; and device placement/transfer evidence on both WX9100s. These are future gates, not work performed by this review.

G0 correctly blocks any source, test, build, runtime, device, dashboard, simulation, evaluation or training work until the plan/campaign/lease/authorization conditions are met. It explicitly protects campaign PID 2750253 and does not claim the gate is satisfied. The plan and this review make no human-approval, implementation, training, or qualification claim.

No blocking Pass 1 finding remains for this exact plan artifact. ACCEPT applies only to the reviewed plan head and permits only the later gates stated in the plan; pending P2/P3/P4 and P1 implementation interfaces still block P5 implementation.

## Review limits

No tests, builds, runtime, GPU, simulator, dashboard, campaign, training, checkpoint, or output operations were performed. This review does not satisfy G0 or authorize any such operation.
