# Independent Pass 3 review: Evaluation/curriculum V4 owner response

- **Verdict:** ACCEPT
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** `/root/closure_wave1_reviewer`, independent exact-head Pass 3 review
- **Response author:** `/root/evaluation_curriculum_owner` (delegated evaluation/curriculum technical owner)
- **Reviewed commit:** `57fe139b9f22f6bfee4bf874119117e26aa8f84c`
- **Reviewed file:** `docs/plans/v4-owner-responses/evaluation-curriculum.md`
- **Reviewed file SHA-256:** `4c4b06740f08b17396824626acee686d13d5ee3c0394538ec5449d244867d6d5`
- **Pinned proposal:** V4 commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- **Pinned matrix:** commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.
- **Pinned owner-closure plan:** commit `4bef4525acf8022f3b38edaafe0793d13ef7a0a2`, SHA-256 `b0a72285269189f383d25e3463a258fa97a692bdee7674792e213eb0ec8657b7`.
- **Supersedes:** prior ACCEPT record at this path for response commit `7bd9135a740b2853c7512102cdcab32e665e5a5b`; this record reviews the revised exact head.

## Review scope and evidence

Reviewed only the revised Evaluation/curriculum row against the exact V4 proposal, owner matrix, closure plan, cited training/counterfactual source, Phase 0 score-ledger provenance, and qualification protocol. The row-local **ADOPT** is supported as a delegated decision that incorporates the supplied user choices. The response clearly distinguishes those choices from a materialized human-approved Phase 0 artifact and from implementation conformance.

The source citations support the bounded current behavior: `_run_macro_step` sums gross simulator reward across executed physics ticks (up to configured macro length); PPO GAE discounts once per macro transition at the configured gamma; route/risk data are carried separately and the D1/D2 comparator subtracts those costs from its utility; and the counterfactual collector uses a 160-physics-tick horizon with 12-tick macros, K=16, `cv`, and one discount exponent per continuation macro. The response explicitly says current on-policy GAE does **not** implement a 160-physics-tick capped target and requires a separate source plan to bind the selected fixed-horizon objective to V4 returns. It does not mislabel the bounded teacher-continuation counterfactual target as the PPO return.

The terminal discussion accurately identifies the current `done = ended or timeout` path and zero bootstrap on `done` as differing from V4's distinct terminal/truncation semantics. It also notes the counterfactual collector combines terminal and timeout and requires an end-cause representation. These are honestly reported blockers, not claims of conformance. The score-estimand section describes only accepted active-HUB FUEL awards in `simulated_score_unit`, preserves null official points and documented exclusions, and states that the source/rules-bound approval artifact and Phase 0 binding are still absent despite the user decision resolving the product choice.

Qualification requirements and remaining gates are consistent with `MECO_QUALIFICATION.md`: target-scoped calibrated v5 protocol, selected matched D1/D2 comparator and separate D0 floor, required category/layout/family episode coverage, checkpoint and prerequisite bindings, metadata-only readiness limitations, and three passing target rounds. The response distinguishes proposed broad curriculum thresholds from the operative qualification protocol and keeps optional bounded recall separate from qualification. Its remaining blocker list accurately states that owner adoption, diagnostic evidence, and metadata preflight do not qualify tracks.

The response has the correct role, exact V4/matrix/closure pins, and row-limited authority. It claims no separate human maintainer approval, other-owner adoption, ABI publication, source implementation, or qualification. Exact response commit and blob SHA-256 were independently rechecked. No tests or runtime actions were performed.

This review accepts the response as an evidence-backed Evaluation/curriculum technical decision for its row only. It does not imply that its source changes or qualification gates are complete.
