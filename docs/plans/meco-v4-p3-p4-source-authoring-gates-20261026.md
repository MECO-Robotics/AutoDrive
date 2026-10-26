# P3/P4 source-authoring gate packet

**Purpose:** Record the first source-only slices authorized by the accepted MECO V4 plans, their exact start gates, and what remains blocked. This packet is planning evidence only. No implementation, integration, tests, builds, device/GPU, simulator, dashboard, campaign, or lease operation is authorized here.

## Current disposition

P0a decisions are accepted, and the P3 package plan has an exact Pass 1 ACCEPT. P3 source authoring is **not executable yet** because its required P1 and P2 candidate API heads with independent SOURCE/API ACCEPT do not exist. P4 is also **not executable**: its latest package-plan candidate has no exact-head Pass 1 ACCEPT, and its required P1/P2/P3 candidate API heads are unavailable.

The current main head `d74b0a15e` does not descend from the accepted implementation source base `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`, accepted overall-plan commit `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, or P0a/P3/P4 plan commits below. Package source worktrees must use the plans' exact approved source base, not current main or a moving branch. Keep planning/review artifacts on their existing refs and record exact pins in the implementation ledger; do not merge/copy them into pre-G0 source worktrees.

## First slices and exact start gates

### P3 Plan A (A1/A2/A3)

**First slice when gates clear:** accepted-input validation and deterministic Plan A support mapping in the standalone adapter. Follow with A1/A2/A3 projection and cost decomposition, device scoring/masking/digest, and only later simulator evidence. Before G0, these are source/test-file authoring plus static API/diff review only.

**Start only when all are recorded:**

1. Accepted P3 plan `2b06304bde63cace9f59b6b25497c82250cf759e`, plan SHA-256 `92a57003e63b0aad90c95f45b23c591106d03071675ff5f58ec39a8367777e60`, with exact Pass 1 ACCEPT record at `9790c4ad253074ff000c2ee98ae178aae58641cd` (record SHA-256 `7f0ae276d1c48ffcd7a320440f8ff73d6b3022c01c2edc40faf0d23f20910b66`).
2. P0a candidate `56183cefd2402342ce28f836155164fc850ebc60`, file SHA-256 `c8ace83e0e5a3fdef8d33ab090d3cbb9193e12342b015d72b07016acdb603726`; exact Pass 3 acceptance `d5dab585ef1147f0d9997ece80db6e9a7ba95979`, review record SHA-256 `25a2c38c502599a4da9c9448bdd3e6850b3ae7145d5977870e17b98304c280e6`.
3. Maintainer decision record `7007607010636fe7451ad1b59eb1a900e0a4d7b1`, file SHA-256 `d9394b07bc508021a4b10faf89b2d258a863183288d633442f796fc181592e64`, recording “Approve all.”
4. Exact P1 candidate API source/interface head and exact P2 candidate API source/interface head, each with independent SOURCE/API ACCEPT and file/blob hashes entered in the ledger. Accepted plans alone do not meet this gate. P2's latest corrected package plan is `09cf081a4b47b66a9b3fcecf3c3748eda5722465`, file SHA-256 `eb034463a5c4d0c655fe8e16d7636247ab899ec56bca1c4655122f725c5041d1`; review record `de260bde20b30368d84531767cc0e8d3bf072b35` accepts that plan, but a P2 implementation/API head is still absent. P1 implementation/API head is also absent.
5. Distinct named P3 implementer and pre-G0 source/API reviewer. Neither may be the P3 planner; final post-G0 reviewer must also be independent.

P3 source-only worktree: new isolated ZFS worktree from `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`; consume exact P1/P2 API pins by reference only. Its complete finite allowlist is `frc_defense/meco_plan_a_policy.py`, `frc_defense/meco_plan_a_v4.py`, `tests/test_meco_plan_a_v4_contract.py`, and `tests/test_meco_plan_a_v4_simulator.py`. Pre-G0 authoring does not permit executing either test file. P3's accepted plan leaves trainer/PPO, P2 producers, shared `meco_tracks.py`, P4, P6, HIP/C++, and all unlisted paths out of scope.

### P4 Plan C (C1/C2)

**First slice when gates clear:** typed fixed-axis row/view contract. Follow with C1/C2 model adapters, common arbiter/D1 parity, residency fixtures, and later seeded simulator evidence. Before G0, source/test-file authoring and static API/diff review only.

**Start only when all are recorded:**

1. Exact-head Pass 1 ACCEPT for the corrected P4 package plan. Latest candidate is `c277571f1e286468445b912f8ce377279f4f4056`, file SHA-256 `b93e87b1fe1f18ef6aea99619f319cf0d91ffe0fa3c1d6747826f8e7cd754a0c`, blob `363d095d56c77f1ded2c8915a993125f4f5bdf17`. It currently says review pending. The prior P4 review `948b3ffd24da9d92b2435d8840dc7d9bf4973ff2` (record SHA-256 `e7e16267633e8eab9ea6810333a2968a2f1ccc3f618e4b6439c98a3eab5112b5`) is **REVISE** on older candidate `257ed8af…`; it is not acceptance of c277571. The correction updates the overall-plan pin, but still requires a fresh exact-head review.
2. The accepted P0a candidate, Pass 3 record and maintainer decision listed above.
3. Exact P1, P2, and P3 candidate API heads, each with independent SOURCE/API ACCEPT and exact file/blob hashes entered in the ledger. The P3 plan candidate or its package ACCEPT does not substitute for a P3 implementation/API head.
4. Distinct named P4 implementer and pre-G0 source/API reviewer; post-G0 final reviewer independent from both.

P4 source-only worktree: fresh isolated ZFS worktree from approved source base `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`, with accepted dependency interfaces pinned by reference only. Its finite allowlist is `frc_defense/meco_plan_c_v4.py`, C1/C2-only adapter/factory hunks in `frc_defense/meco_tracks.py` if needed, `tests/test_meco_plan_c_v4_contract.py`, and `tests/test_meco_plan_c_v4_simulator.py`. Pre-G0 authoring does not permit test execution. P2 candidate/route producers, P3 Plan A, P5 trainer/PPO/RNG/checkpoint, P1 artifact/schema, P6 dashboard/qualification, HIP/C++, and all unlisted paths remain out of scope.

## Deferred gates and frozen decisions

- G0 remains closed: the user must manually stop diagnostic campaign PID `2750253`; both WX9100 leases must be independently confirmed released; maintainer authorization for execution/integration must be recorded; and the post-G0 contract-anchor bridge must receive its required review before integration/validation. Do not inspect or touch the live campaign, devices, leases, workers, outputs, checkpoints, or dashboard.
- No build/compile, test execution, import/runtime execution, GPU/device use, simulator/numerical work, calibration/evaluation, dashboard/replay work, integration, merge, promotion, training, or qualification is authorized before G0.
- P0a approves the score proxy, separate `pickup_simple` Stage 1 and `full_match_offense` score protocols, case matrices, and proposed analysis/safety/failure rules. Exact expanded case/seed bytes and frozen opponent policy/checkpoint/config identities remain source-bound artifacts to freeze before any new candidate training or outcome collection; P0a does not invent them.
- Source-only acceptance is package-specific. A candidate API SOURCE/API ACCEPT permits only the dependent source authoring described in its accepted plan; it is not final Pass 3 acceptance or runtime/integration authority.

## Provenance and review limits

Accepted overall staged-gate plan: `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, file SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`, Pass 1 record `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`. P4's plan itself cites this accepted overall-plan pin. This packet does not update the P3/P4 plans or grant either package more authority than their exact reviewed plans.

Review performed against Git objects and planning/review records only. No runtime source was edited. No tests, builds, simulator, dashboard, GPU/device, campaign, or lease activity was performed.
