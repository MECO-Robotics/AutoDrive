# Independent Pass 1 review: V4 schema, score-event, and artifact implementation plan

- **Verdict:** REVISE (supersedes prior Pass 1 REVISE at `c92480036`)
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** fresh exact-head Pass 1 review assigned by `/root`
- **Plan author:** `/root/plan_c_owner`
- **Reviewed plan commit:** `2e77c283addd6ec6b508065b8378e392fba2ed87`
- **Reviewed file:** `docs/plans/meco-five-track-v4-schema-score-artifact-implementation-plan-20261026.md`
- **Git blob:** `7db05501b09f6cc722434bd5bd0e3776d384377a`
- **Plan file SHA-256:** `041fd82579217aa3628ce23d980ebf3b6e586d51a8c96fba9238fc6970f37e6f`
- **Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted overall plan:** commit `6325920252773cfa05d63167ba3c29a90debdb01`, file SHA-256 `9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a`; exact Pass 1 ACCEPT record commit `be845d7fb2d2ece861dcd158ac8b8dd89e3dc59d`, record SHA-256 `5a6e94f978c8a7e18d9623897ceba6f32f5b5eadb0aba6605f21c9d1e941a886`.
- **Accepted P0 package plan:** commit `13dbd7f8f953995850e811202f7e9c5264c49dca`, file SHA-256 `09ec722d64d01d99b60cc9ed52e47ce68790e548f66107cc1a19dd9a7a588adf`; exact Pass 1 ACCEPT record commit `037f0f9386cc0efadb1fb54efe5f38f097926d0e`, record SHA-256 `1ee70a6e9d24dc5201da5eec446820167cb8c4d4436fbbe013eb16de66b0280d`.
- **Accepted reconciliation:** commit `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; accepted reconciliation review commit `5cb1a2d525168ff0443089cf329e66c6ecb917b8`, record SHA-256 `0df700dcaef1ed35672efc468bdcfd78fac829c31bcf7a88b3c51f1951f06c32`.
- **Reconciled profile hash:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`
- **V4 proposal/profile digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`

## Review scope and evidence

The corrected plan now pins the accepted top-level P0a/P0b amendment and accepted P0 scoped package plan, with the exact file hashes and review records. The reconciled profile and raw proposal digest remain distinct and correctly pinned. The P0a/P0b correction resolves the prior circularity: P1.1 source discovery depends on G0 and accepted P0a; it supplies the P0b source/rules artifact; P1 implementation and regression validation use P0a and do not wait for P0b; P0b occurs after P1–P6 source freeze and gates fresh training and P6 runtime qualification. P0b does not depend on P6 runtime results. P0b is generated-output-only, while P6 owns the qualification schema, runner, and qualification source/tests.

The plan retains a finite allowlist, a source-discovery gate that does not invent the absent score-ledger module, source-backed event identity/producer/consumer requirements, exact schema/event capacities and fail-closed behavior, two-WX9100/no-CPU-fallback validation, and no-touch campaign boundaries. It incorporates P0a preregistered seeds for P1 behavior checks and the AGENTS.md requirement for seeded end-to-end simulator/replay inspection and current dashboard backend/frontend verification. The missing source map, P0b calibration, implementation tests, device evidence, simulator results, and dashboard checks are correctly future gates, not claimed as done.

## Blocking finding: allowlist conflicts with exclusive P5 trainer ownership

The P1 exact implementation allowlist authorizes edits to `frc_defense/meco_track_training.py` for V4 batch packing and manifest integration. The accepted overall plan's P5 package makes P5 the **sole owner** of edits to this shared trainer file. The P1 plan's concurrency note cannot reassign that ownership locally: it says the coordinator may reserve the file to P1 or defer conflicting edits, but the accepted top-level plan has already reserved it to P5. This is a direct package-scope conflict and leaves the transition-row/artifact trainer boundary ambiguously owned.

Remove `frc_defense/meco_track_training.py` from P1's edit allowlist and make P1 deliver the typed batch/manifest API and producer/consumer contract. Assign trainer-side consumption and batch-packing integration to P5, or request and independently review an amendment to the top-level package ownership before P1 begins. Align P1.4's deliverable and P5's dependency on that interface, then submit a fresh exact-head Pass 1 review.

The plan's sequencing and technical pins are otherwise satisfactory, but this exact head is **REVISE** until shared-file ownership agrees with the accepted top-level plan. No implementation or validation activity is authorized by this review.

## Review limits

This REVISE applies only to the exact P1 plan artifact and commit pinned above. It does not authorize source changes, tests, builds, simulator/runtime/device work, dashboard actions, training, or qualification. No source, tests, GPU, simulator, dashboard, or campaign actions were performed during review.
