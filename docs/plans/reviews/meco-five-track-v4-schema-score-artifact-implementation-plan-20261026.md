# Independent Pass 1 review: V4 schema, score-event, and artifact implementation plan

- **Verdict:** ACCEPT (supersedes prior REVISE records for `c9d37b43bbddaa0984665dd918a22c572a5cc59d` and `2e77c283addd6ec6b508065b8378e392fba2ed87`)
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** fresh exact-head Pass 1 review assigned by `/root`
- **Plan author:** `/root/plan_c_owner`
- **Reviewed plan commit:** `53686a254318484f9c6bc3496ecc1c461572c53a`
- **Reviewed file:** `docs/plans/meco-five-track-v4-schema-score-artifact-implementation-plan-20261026.md`
- **Git blob:** `39e313f28228887fd2ed3c71f6b0d290ae97423d`
- **Plan file SHA-256:** `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076`
- **Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted overall plan:** commit `6325920252773cfa05d63167ba3c29a90debdb01`, file SHA-256 `9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a`; exact Pass 1 ACCEPT record commit `be845d7fb2d2ece861dcd158ac8b8dd89e3dc59d`, record SHA-256 `5a6e94f978c8a7e18d9623897ceba6f32f5b5eadb0aba6605f21c9d1e941a886`.
- **Accepted P0 package plan:** commit `13dbd7f8f953995850e811202f7e9c5264c49dca`, file SHA-256 `09ec722d64d01d99b60cc9ed52e47ce68790e548f66107cc1a19dd9a7a588adf`; exact Pass 1 ACCEPT record commit `037f0f9386cc0efadb1fb54efe5f38f097926d0e`, record SHA-256 `1ee70a6e9d24dc5201da5eec446820167cb8c4d4436fbbe013eb16de66b0280d`.
- **Accepted reconciliation:** commit `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; accepted review record commit `5cb1a2d525168ff0443089cf329e66c6ecb917b8`, record SHA-256 `0df700dcaef1ed35672efc468bdcfd78fac829c31bcf7a88b3c51f1951f06c32`.
- **Reconciled profile hash:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`
- **V4 proposal/profile digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`

## Review scope and evidence

The exact P1 plan pins the accepted top-level P0a/P0b amendment, accepted scoped P0 package, source base, and reconciliation/profile artifacts. The P0 sequencing correction is sound: P1.1 read-only source discovery depends on G0 and accepted P0a; it provides P0b with the source map and identity inputs; P1 source work and regression validation do not depend on final P0b; P0b follows the reviewed P1–P6 source freeze and gates fresh training and P6 runtime qualification, not P1 implementation or tests. P0b is not cyclic with P6 runtime qualification.

The latest correction resolves the previously reported shared-file ownership conflict. P1 now exposes a standalone GPU-resident batch and manifest API from its own producer/artifact scope, explicitly excludes `frc_defense/meco_track_training.py` and all trainer integration tests, and assigns caller integration, optimizer/RNG/checkpoint no-mutation tests, and shared trainer edits to P5. P1.4's output is an exact API head that P5 consumes after independent acceptance. This matches the accepted top-level plan's exclusive P5 ownership of the trainer.

The rest of the source scope is finite and consistent with P1's responsibility: typed V4 schema/reader and event modules, source-proven capture at actual score mutation sites, event/reset buffers and typed simulator output seam, plus focused schema/event/source-site fixtures. The absent score-ledger module remains explicitly outside the allowlist; P1.1 must search the exact base and stop for amendment if resolution requires another path. The plan preserves exact profile/event capacities, byte counts, version-first exact-length reader behavior, raw proposal digest semantics, source identity, and fail-closed handling. It does not expand into PPO/GAE, curriculum, model behavior, or trainer integration.

P1.5 requires tests that exercise simulated scoring to be paired with a representative seeded end-to-end simulator evaluation and replay inspection, using P0a-frozen scenarios/seeds. It also requires restarting and verifying current dashboard services/backend/frontend/endpoints before fresh replays, and recording provenance. Algorithmic score/event/digest work is assigned to both WX9100 devices with no CPU/P600 fallback; G0 protects the active campaign and leases. These are future package gates; none were performed in this review.

No blocking Pass 1 finding remains for this exact P1 plan. Its ACCEPT authorizes only the plan's own future gates; it does not satisfy G0/P0a, begin source work, authorize devices/tests/dashboard/simulation, or permit P5 trainer integration before P1's exact API head is accepted.

## Review limits

This ACCEPT applies only to the exact plan artifact and commit pinned above. No source, tests, runtime, GPU, simulator, dashboard, campaign, or training actions were performed during review.
