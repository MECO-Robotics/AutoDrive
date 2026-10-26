# Independent Pass 1 review: P1 V4 schema and score-event source package

- **Verdict:** REVISE
- **Review date:** 2026-10-26
- **Planner:** `/root/p1_p2_planner`
- **Reviewed commit:** `376f3913976839a1c6043915c7b7854d449c921b`
- **Reviewed file:** `docs/plans/meco-v4-p1-score-event-source-package-plan-20261026.md`
- **Git blob:** `68af1d18da12d7b1bce4e0f4e5eb914c859bde2c`
- **Plan SHA-256:** `b0c25979f1bfe8ab43f9d1ff784c40fe6e794f71df2b42ec363b53e5091d6571`
- **Review base:** `d74b0a15e25ddbc6bacfbbb864674371629bb19e`
- **Scope:** exact source base, finite allowlist, successor relationship, P1.1 source-semantic gate, P5 ownership seam, and pre-G0/G0 boundaries. Static Git/document review only.

## Review evidence

The candidate correctly identifies itself as a finite successor to accepted P1 plan `53686a254318484f9c6bc3496ecc1c461572c53a` and preserves that plan's authority until this successor is accepted. It pins the current accepted overall plan `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, the approved implementation base `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`, accepted P0a candidate/maintainer decision/Pass 3 record, and the reconciled ABI/profile. The source-file hashes listed in the source map match the exact pinned base. The finite allowlist is explicit, retains the source-map deliverable and bounded event/schema files, excludes the shared P5 trainer, and requires a reviewed amendment for any unlisted dependency.

The P1.1 gate is conservatively bounded. The plan distinguishes fixed ABI bytes from source semantics that the base does not establish: physical-slot generation lifecycle, episode event sequence/reset/order, tick/elapsed-time meaning, and the absent-at-base score ledger. It permits a source-map resolution only from the pinned source and accepted contracts; otherwise it requires a stop and reviewed amendment before event-path code. In particular it does not authorize choosing generation or sequence conventions by implementer judgment. P1.3 depends on P1.1 closure. The API handoff keeps trainer rollout, reward/counterfactual collection, PPO/RNG, and checkpoint integration under P5 and requires an exact pre-G0 SOURCE/API ACCEPT before P5 authoring.

The pre-G0 source/API review and prohibition on execution are explicit. However, the post-G0 validation gate below is incomplete relative to the accepted overall plan.

## Finding

### F1 — Post-G0 validation omits required G0 authorization and contract-anchor review

The accepted overall plan's Gate G0 requires, in addition to manual stop evidence for campaign PID `2750253` and independent release evidence for both WX9100 leases, explicit user or maintainer authorization for runtime/device/integration work. It further requires the reviewed docs-only contract-anchor integration after G0 and **before any validation or integration** (accepted overall plan, lines 171–203).

The candidate's “Post-G0 package acceptance” section permits package validation once the campaign has been manually stopped and both leases are independently confirmed released. It does not require the separate maintainer authorization or reviewed contract-anchor bridge first. Its phrase “Only after” those two conditions is therefore an incomplete G0 gate and could authorize tests, simulator, dashboard, and device validation before the remaining required gates.

Amend the validation/runtime gates to require all recorded G0 evidence, including explicit maintainer authorization, and the exact reviewed contract-anchor bridge before any validation, device/runtime work, or integration. Preserve the source-only pre-G0 allowance and its prohibition on execution. Request a fresh exact-head Pass 1 review after the correction.

### F2 — Two-WX9100 residency and forbidden fallback are not explicit

The accepted overall plan requires algorithmic numerical work on both assigned WX9100 devices, forbids CPU numerical fallback/P600 substitution/silent migration, and limits CPU use to orchestration and opaque I/O. The accepted predecessor P1 plan also explicitly required score-event numerical simulation/capture on each assigned WX9100 and no CPU/P600 score or digest work (predecessor plan, lines 298–301, 377). The successor correctly says dynamic event validation and digesting are device-side, but its post-G0 acceptance asks only to record “device UUID/runtime” and does not require evidence for both devices or explicitly fail closed on CPU/P600 fallback or migration.

Carry the two-device residency requirement into this successor: run the P1 algorithmic path on each assigned WX9100, record both UUID/runtime and relevant placement/transfer evidence, and prohibit CPU/P600 computation or silent migration. CPU may remain limited to process supervision, configuration, and opaque persistence. The plan must require an explicit failure if either approved device or required kernel path is unavailable.

## Review limits

This REVISE applies only to the exact candidate commit and hash above. It does not authorize source edits, test execution, builds, imports, simulator, dashboard, GPU/device use, campaign actions, integration, merge, training, or qualification. No operational validation was performed.
