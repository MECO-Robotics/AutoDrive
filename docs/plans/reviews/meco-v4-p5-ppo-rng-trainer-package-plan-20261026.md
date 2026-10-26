# Independent Pass 1 review: P5 PPO/RNG trainer package plan

- **Verdict:** REVISE (supersedes prior ACCEPT records for earlier P5 heads; this is a new gate-amendment head)
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** independent exact-head review assigned by `/root`
- **Plan author:** `plan_a_owner`
- **Reviewed plan commit:** `0e8fbcad395a4cdcefc53c90d1013072b94ca3f8`
- **Reviewed file:** `docs/plans/meco-v4-p5-ppo-rng-trainer-package-plan-20261026.md`
- **Git blob:** `c574e0092f16f079e2e4078bae56468691983981`
- **Plan file SHA-256:** `4180ea83e50c8cf352e8f2c2216473a14cf3232cb614917b3d13d7596c3b433d`
- **Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Latest accepted overall staged-gate plan:** commit `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; Pass 1 ACCEPT record commit `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`
- **Accepted P0 staged-gate plan:** commit `c4f136b267f8e41e0baec53394f28fac28f16eb5`, file SHA-256 `ef989468059b16fc184d6a7740ad8dfa07db776b0f5dd9a9d220525f25a491b2`; ACCEPT review commit `a65a031cad69bc4536c33b68b3839377d2987d2a`, record SHA-256 `4ab531d06f2866f75e494c0cc20b0be40c4c9b3c38e992a06ab541f98868a9f0`
- **Accepted P1 staged-gate plan:** commit `606ba5adce04afd84bfb372260aebde8fd4fcec6`, file SHA-256 `7dc5be35d3826a6047653021e97a941d6872a73a6997ba768e95753f693b0653`; ACCEPT review commit `77816edbd1923c018d4af774b2f1b3bae0f415b8`, record SHA-256 `3dbadfb7243d891fdacf993962722b90e245120442c039a15a48def97f0ca193`
- **Accepted P2 staged-gate plan:** commit `09cf081a4b47b66a9b3fcecf3c3748eda5722465`, file SHA-256 `eb034463a5c4d0c655fe8e16d7636247ab899ec56bca1c4655122f725c5041d1`; ACCEPT review commit `de260bde20b30368d84531767cc0e8d3bf072b35`, record SHA-256 `fe66a8eccb71d7be58d6cc37a20e98f32132ce36f03fa0f09ec0aa4ec5370d18`
- **P3 staged-gate plan:** commit `c2094a60484efa58e7e4b0a626398dcae71e665b`, file SHA-256 `3748f5ae60aabb20b5b62bc3fe37a5efd71dfc69d87ff4079f47a277d34cf611`; exact-head review returned REVISE at record commit `baa7f1d831edaa0bcdf81548709cce51fab23d77` (overall-plan acceptance status pin must be corrected)
- **Accepted P4 staged-gate plan:** commit `c277571f1e286468445b912f8ce377279f4f4056`, file SHA-256 `b93e87b1fe1f18ef6aea99619f319cf0d91ffe0fa3c1d6747826f8e7cd754a0c`; ACCEPT review commit `1ca6c8f4e33f82a3c89ec76c2897336006a9f751`, record SHA-256 `af215ee0fe1388d3f00e6e21849b3ab34d165ec33cc04a14a1189c68e5092cc4`

## Review scope and evidence

The staged execution gates are coherent. P5 may author only its finite source/test allowlist in an isolated source-base worktree after exact candidate APIs receive SOURCE/API ACCEPT. It forbids integration or runtime use before G0. G0 requires manual stop of PID `2750253`, independent release confirmation for both WX9100 leases, explicit maintainer authorization, and the reviewed post-G0 contract-anchor bridge before validation/integration. Tests, builds, simulator, GPU/device, dashboard, training, checkpoint mutation, and qualification remain blocked before G0. P5's source-authoring gate does not wait on P0b; fresh training and P6 qualification do.

P5 technical scope remains unchanged: P5 alone owns shared trainer and counterfactual integration, PPO/GAE/end-cause semantics, Philox streams, atomic update/checkpoint behavior and the listed tests; P1 owns schema/event/artifact bytes; model, environment and qualification work remain assigned to their other packages. The accepted 160-tick horizon, 12-tick macro cap, gamma 0.993, terminal/truncation/bootstrap semantics, GPU-only algorithmic work on both WX9100 devices, no CPU/P600 fallback, seeded simulator/replay and current-dashboard requirements remain explicit. No P5 implementation or validation is claimed.

### Finding: refresh dependency status and exact accepted plan pins

The current dependency/status entries are stale against the exact plan heads and reviews now controlling this sequence:

- The overall `8efec...` plan entry says its independent review is pending, but it has exact Pass 1 ACCEPT record `b0b2962...` (and hash above).
- The P0 and P2 gate amendments are marked pending fresh review, but their accepted exact heads and review records are now `c4f136b...` / `a65a031...` and `09cf081...` / `de260bde...`.
- The P1 dependency still cites the earlier `53686a...` plan, although the staged-gate successor `606ba5...` has been accepted.
- P4 is described as “in progress; no accepted plan” even though the staged-gate plan `c277571...` has an exact ACCEPT record. The P4 implementation/API remains future and should be stated separately.
- P3's gate amendment is not merely pending fresh review; exact head `c2094a...` received REVISE at `baa7f1d...` and is awaiting correction. The base P3 plan remains accepted, but the gate amendment state should be represented accurately.

Update P5's status table/header to pin the current exact plan heads and review records, distinguishing package-plan acceptance from future candidate API/SOURCE/API ACCEPT and integration heads. Preserve the current future API gates; these updates do not mean P5 may start implementation. Until the status ledger is corrected, the plan does not provide a reliable current dependency map for Pass 2.

## Review limits

This review applies only to the exact P5 plan commit, blob, and file hash above. No source, tests, runtime, GPU, simulator, dashboard, campaign, or training actions were performed.
