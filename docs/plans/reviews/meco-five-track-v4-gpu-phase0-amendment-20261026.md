# Independent Pass 1 review: top-level V4 Phase 0 sequencing amendment

- **Verdict:** ACCEPT
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** independent Pass 1 amendment review assigned by `/root`
- **Plan author:** `plan_a_owner`
- **Reviewed plan commit:** `5d02eab99738cefa0318d4af4018c0450ecca685`
- **Parent commit:** `223ac4a3f3c657193ef5623b383716364a9395d5`
- **Reviewed file:** `docs/plans/meco-five-track-v4-gpu-implementation-plan-20261026.md`
- **Git blob:** `d9fd85d2c2050848479774876d9a367bfae313cf`
- **Plan file SHA-256:** `c7e6d12c4c9e6c9d48d05015e0eb8eac91fe9b42c26e49a52fb81e52c1a4a37c`
- **Pinned implementation base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted reconciliation:** `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, file SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; accepted review commit `5cb1a2d525168ff0443089cf329e66c6ecb917b8`, review-record SHA-256 `0df700dcaef1ed35672efc468bdcfd78fac829c31bcf7a88b3c51f1951f06c32`.
- **Reconciled profile hash:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`
- **V4 envelope profile digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`

## Review scope and evidence

The amendment is a direct child of the previously accepted top-level plan commit and changes only that plan document. It preserves the exact source base and accepted reconciliation pins. The amendment separates the former P0 into (a) P0a preregistration of the score estimand/rules, v5 protocol schema, disjoint cases/seeds, and calibration/margin method before P1–P6 implementation; and (b) P0b final source-bound calibration and immutable artifact after P1–P6 source is frozen and independently reviewed.

The sequencing removes the prior circular dependency. P0a states the intended score semantics and explicitly assigns source-site verification to P1's read-only discovery before implementation. P1–P6 source work depends on accepted P0a. P0b depends on frozen/reviewed P1–P6 source, including the P6 qualification runner, but expressly does not depend on P6 runtime qualification results. P6 runtime qualification requires accepted P0b, integrated/reviewed P1–P6, fresh source-bound checkpoints, dual-device readiness, and separate human authorization. Thus the dependency path is P0a → P1–P6 source freeze → P0b → P6 runtime qualification, without a P0/P6 runtime cycle.

The calibration/source identity gate is sufficiently defined: P0b recomputes source/rules/config identity from the exact frozen integration commit; verifies the actual producer/consumer mapping and source-bound metric code against P0a; runs only preregistered D0 cases, partitions, and repeats on the two WX9100 devices; retains records for independent recomputation; binds the final source, profile, devices, score rules, cases/seeds, evidence, margins, and approval provenance; and invalidates the binding if relevant post-freeze source changes. P0a fixes the estimand, case/seed bytes and derivation, and calibration/margin method before candidate outcomes. Human approvals remain explicit gates and are not inferred from hashes or agent-authored records.

G0 and campaign/resource boundaries remain intact. No implementation, tests, build, runtime, device, dashboard, calibration, simulation, training, or qualification work begins before plan acceptance, user-stopped campaign evidence, independently released GPU leases, and explicit authorization. Algorithmic work, including final calibration and qualification, remains assigned to the two WX9100 GPUs with no CPU numerical fallback. Package ownership retains distinct planner, implementer, and reviewer roles. The five tracks and three-round Stage 1 criteria remain in scope. The existing AGENTS.md seeded simulator evaluation and current-dashboard replay requirements remain stated as future package validation gates; no such validation was performed in this review.

One source wording is read in context with the plan's explicit distinction between P0a preregistered rules/case-generation inputs and P0b's final code/source identity: implementation proceeds under P0a, while any behavior-changing discrepancy is returned for preregistration/reconciliation before candidate results, and final calibration binds the frozen implementation. That interpretation is supported by the P1 source-discovery requirement and P0b's source verification and invalidation gates.

## Review limits

This ACCEPT applies only to the exact plan document and commit pinned above. It accepts the sequencing amendment as a planning artifact; it does not authorize implementation, tests, builds, device/runtime work, dashboard actions, calibration, training, evaluation, ABI publication, or qualification. Those remain behind the stated package and human authorization gates. No tests, runtime, GPU, dashboard, simulator, or campaign actions were performed during review.
