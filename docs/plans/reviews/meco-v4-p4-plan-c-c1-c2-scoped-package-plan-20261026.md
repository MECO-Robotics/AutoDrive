# Independent Pass 1 review: P4 Plan C C1/C2 scoped package plan

- **Verdict:** ACCEPT
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** independent exact-head Pass 1 review assigned by `/root`
- **Plan author:** `plan_c_owner`
- **Reviewed plan commit:** `8708ba8a96536f1a3ae28133afcd4ebcb4790378`
- **Reviewed file:** `docs/plans/meco-v4-p4-plan-c-c1-c2-scoped-package-plan-20261026.md`
- **Git blob:** `802f3e555fd43e5a149e6f693fbfcebb3821b5c0`
- **Plan file SHA-256:** `e611a87f4b257477f81241fbc67b3b967c59da600de8e26d696d6dbfacd47b1f`
- **Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted overall plan:** commit `6325920252773cfa05d63167ba3c29a90debdb01`, file SHA-256 `9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a`; Pass 1 ACCEPT record commit `be845d7fb2d2ece861dcd158ac8b8dd89e3dc59d`, record SHA-256 `5a6e94f978c8a7e18d9623897ceba6f32f5b5eadb0aba6605f21c9d1e941a886`.
- **Accepted P0 plan:** commit `13dbd7f8f953995850e811202f7e9c5264c49dca`, file SHA-256 `09ec722d64d01d99b60cc9ed52e47ce68790e548f66107cc1a19dd9a7a588adf`; Pass 1 ACCEPT record commit `037f0f9386cc0efadb1fb54efe5f38f097926d0e`, record SHA-256 `1ee70a6e9d24dc5201da5eec446820167cb8c4d4436fbbe013eb16de66b0280d`.
- **Accepted P1 package plan:** commit `53686a254318484f9c6bc3496ecc1c461572c53a`, file SHA-256 `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076`; Pass 1 ACCEPT record commit `854d5286947074ed33f7ac059d67f2f9ccbf9257`, record SHA-256 `0680a57e5ff0427b20a0a5bd86c224085583b241b3f57bcb18bde69cab0f158f`.
- **Accepted P2 package plan:** commit `8d0727f10d46bfc5fee2810a1ec7fa7987ecaf91`, file SHA-256 `5eddd42914ab74426257ab8c1ff883b25c0c39aaa40bf7c40ccc2e363fbb3957`; Pass 1 ACCEPT record commit `0292f01638bf3180b4952721f5692a4446841013`, record SHA-256 `ad43d8e9c91415eb08fabdced1ba85aabaf0849dc15cc302c8f2965c4c0b31bb`.
- **Accepted P3 package plan:** commit `8b641ccc56b7c73a880a095b3c6940cb03b617f3`, file SHA-256 `6958cb883a339a251e6069378d8ebb853be97e4965d30ca0eba8d3003fbc0a9b`; Pass 1 ACCEPT record commit `a11e68b6240c6701a14d6fafe932f65f0b05d6e2`, record SHA-256 `8da034c3d3b01b1bf113fb429f4fc0f3ed7fa708d19279d266e71ca61918e685`.
- **Accepted reconciliation:** commit `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; canonical profile hash `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e`; V4 proposal digest `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- **Accepted Plan C owner response:** commit `6106e2d236af4a0891d8d72a578ce748aece5743`, file SHA-256 `4bad45f3bb5dcbe87907f666805166364a9f424b2fd95a59b1dd23ef3c93c610`; Pass 3 ACCEPT record SHA-256 `cb16ab85da0cd4778c5fe09042ccc3150ece64139a7ed76dee6d7e5b07c3f1a4`.

## Review scope and evidence

The exact contract in the plan matches the adopted Plan C response and reconciliation: 537 stable policy source rows, 20 raw features plus 20 missingness indicators, 40 model columns, `K=16`, 9,681 ordered route sinks, 152 chunks of up to 64, and 73 path points. It spells out the 512–535 variant-major action rows, continuation row 536, separate oracle sentinel domain, descriptor fields and stable identity codes. It specifies the ordinary and continuation maps, selector-local rows, rank and validity masks, finite-zero padding, sentinel behavior, and complete route-sink retention. These mappings agree with the accepted Plan C owner response, which was corrected and reviewed against the owner source mapping before ADOPT.

The policy design preserves a shared candidate universe and row/command identity for C1, C2 and deterministic D1. C1 uses engineered candidate/global/recurrent inputs; C2 adds raster encoding and candidate-to-map attention. Both expose gross utility to a shared deterministic arbiter that applies route and risk costs once, preserves feasibility, fallback and hysteresis, and leaves native execution safety authoritative. D1's deterministic scoring/ordering is treated as source-bound and is not redefined. No Plan A traversal penalty is added.

The finite allowlist is bounded to the standalone Plan C model/scoring adapter, C1/C2-specific portions of `meco_tracks.py`, and two scoped test files. It excludes P2 candidate/observation/route producers, P1 schemas/artifacts, P3 Plan A, P5's shared trainer and P6 qualification/dashboard. The dependency table correctly distinguishes accepted P1/P2/P3 package plans from still-required exact implementation/API heads and defines the required handoffs; P5 remains the exclusive owner of `meco_track_training.py`. The plan explicitly blocks implementation until these exact APIs and G0/P0a are accepted and pinned, and requires a fresh worktree and distinct planner, implementer and reviewer.

GPU-only gates require both assigned WX9100 devices, kernel/device and transfer evidence, CPU/P600 rejection, and no dynamic host computation. The plan pairs behavior tests with representative seeded end-to-end simulator evaluations and inspected replay records; it also requires restarting and verifying the current dashboard backend/frontend/services/endpoints before creating and inspecting a fresh replay. G0 protects the current campaign and leases; the plan claims no stop or release evidence and authorizes no source, tests, runtime, GPU, dashboard or campaign activity.

No blocking Pass 1 finding remains for this exact plan. Accepted package plans do not constitute accepted implementation APIs; P4 implementation remains gated on the exact P1/P2/P3 implementation interfaces and reviews.

## Review limits

This ACCEPT applies only to the exact plan artifact and commit above. It does not accept P4 implementation, satisfy G0/P0a, authorize source changes, tests, simulator/dashboard/device work, campaign changes, training or qualification, or establish current Plan C conformance. No source, tests, runtime, GPU, simulator, dashboard, campaign or training actions were performed during this review.
