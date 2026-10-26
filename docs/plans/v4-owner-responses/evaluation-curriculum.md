# V4 evaluation and curriculum owner response

- **Owner role:** `evaluation_curriculum_owner`
- **Delegated task identity:** `/root/evaluation_curriculum_owner`
- **Response author:** delegated technical owner agent for this scoped row
- **Response date:** 2026-10-26
- **Disposition:** **PENDING**

## Pinned artifacts

- V4 proposal: commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256
  `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- V4 owner-adoption matrix: commit
  `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256
  `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.

## Decision and evidence

I cannot adopt the exact V4 proposal yet. Its row schema gives `reward` and
`macro_reward[12]` fields, and it distinguishes true terminal from truncation
for bootstrap handling, but it explicitly leaves the product estimand to the
curriculum/simulator owners (V4 §“Exact dynamic row field inventory,” lines
157, 160, and §“Codes, lifecycle, and deterministic replay recommendations,”
lines 281–286). It does not select whether strategic gross reward is measured
over a fixed horizon or until the next tactical decision, nor fully specify
discounting and the mapping from macro rewards/outcomes to the evaluation
estimand. These are material because the source learning-track plan requires
executed rollout reward and risk penalties for RL and explicitly leaves the
gross-reward horizon unresolved ( `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`
§§2, 5, and 9, lines 55, 67, 102–104, and 155).

The approved qualification protocol supplies promotion gates, but does not by
itself settle the PPO training estimand. It requires a target-scoped calibrated
v5 protocol, a frozen matched D1/D2 reference, separately reported D0 safety,
and a Phase 0 score-estimand review in the digest-bound evidence index
(`docs/MECO_QUALIFICATION.md` lines 5–15, 275–285, 359–369). Stage 1 then
requires at least 100 episodes per category and frozen layout/family stratum,
paired learned/reference seeds, safety comparison to D0, and three consecutive
passing target rounds under the same protocol (same file, lines 361–369,
388–396). Those are the applicable curriculum gates; a diagnostic reward or
recall metric cannot replace them.

### Recall metric position

For Plan C's optional candidate-recall diagnostic, the source contract supports
the following bounded metric definition: numerator is the number of distinct,
valid, route-feasible selected candidate identities present in the independent
expanded-goal reference pool; denominator is the number of distinct,
route-feasible identities in that pool. The pool is bounded and snapshot-based,
not a simulator-optimality oracle (`frc_defense/meco_candidates.py` lines
1670–1701; `docs/MECO_ABLATION_MATRIX.md` lines 241–248). An empty reference
pool currently yields `1.0`; the recorded zero denominator must remain visible
so this vacuous value is not represented as observed coverage (`meco_candidates.py`
lines 1678–1701). Report separately by track, category, and K, and do not use
recall as a Stage 1 pass criterion. V4 itself says recall is optional and
asserts no recall result (V4 lines 303–319).

## Exact unresolved fields

Before this owner row can adopt V4, a reviewed Phase 0 score-estimand decision
must pin all of the following and link them to the v5 protocol identity:

1. Training reward target: fixed strategic horizon or next tactical decision,
   including discounting, macro-reward aggregation, risk penalties, and how
   terminal and truncation affect the target.
2. Evaluation estimand: exact match-outcome/useful-throughput measure and its
   episode aggregation, with D0 safety remaining separate and the chosen
   matched D1/D2 reference frozen before held-out evaluation.
3. Recall serialization: preserve reference capacity, actual denominator,
   selected/reference identity intersection, and the empty-reference flag so
   the existing `1.0` empty-pool convention is unambiguous. Keep it diagnostic
   and outside qualification gates.

The existing source and documents establish candidate recall's bounded
reference semantics and the v5 promotion procedure, but do not establish the
missing reward horizon/discount/estimand linkage. I therefore leave the whole
owner disposition **PENDING** rather than adopting only part of the row. No
replacement artifact is proposed: these unresolved fields require the Phase 0
score-estimand owner evidence, not a guessed ABI value.

## Scope statement

This records only the delegated evaluation/curriculum technical decision for
this row. It is not human maintainer approval, adoption by another owner,
cross-owner V4 adoption, ABI publication, or authorization for implementation,
tests, runtime, simulator evaluation, dashboard work, campaign changes, or
training.
