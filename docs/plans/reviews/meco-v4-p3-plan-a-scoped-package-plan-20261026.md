# Independent Pass 1 review: P3 Plan A A1/A2/A3 scoped package plan

- **Verdict:** ACCEPT
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/plan_c_owner`
- **Reviewer task:** independent exact-head Pass 1 review assigned by `/root`
- **Plan author:** `/root/closure_wave1_reviewer` (as declared by the plan)
- **Reviewed plan commit:** `8b641ccc56b7c73a880a095b3c6940cb03b617f3`
- **Reviewed file:** `docs/plans/meco-v4-p3-plan-a-scoped-package-plan-20261026.md`
- **Git blob:** `05a03327d2ce0e609484a7087b610e4999f8de9e`
- **Plan file SHA-256:** `6958cb883a339a251e6069378d8ebb853be97e4965d30ca0eba8d3003fbc0a9b`
- **Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted overall plan:** `6325920252773cfa05d63167ba3c29a90debdb01`, file SHA-256 `9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a`; Pass 1 ACCEPT record commit `be845d7fb2d2ece861dcd158ac8b8dd89e3dc59d`, record SHA-256 `5a6e94f978c8a7e18d9623897ceba6f32f5b5eadb0aba6605f21c9d1e941a886`.
- **Accepted P0 package plan:** `13dbd7f8f953995850e811202f7e9c5264c49dca`, file SHA-256 `09ec722d64d01d99b60cc9ed52e47ce68790e548f66107cc1a19dd9a7a588adf`; Pass 1 ACCEPT record commit `037f0f9386cc0efadb1fb54efe5f38f097926d0e`, record SHA-256 `1ee70a6e9d24dc5201da5eec446820167cb8c4d4436fbbe013eb16de66b0280d`.
- **Accepted P1 package plan:** `53686a254318484f9c6bc3496ecc1c461572c53a`, file SHA-256 `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076`; Pass 1 ACCEPT record commit `854d5286947074ed33f7ac059d67f2f9ccbf9257`, record SHA-256 `0680a57e5ff0427b20a0a5bd86c224085583b241b3f57bcb18bde69cab0f158f`. The P1 implementation/API head remains a required future exact-head dependency.
- **P2 dependency status:** no accepted P2 interface head is pinned by the reviewed plan; it accurately marks that interface as a blocking prerequisite. The then-current P2 plan candidate is `8d0727f10d46bfc5fee2810a1ec7fa7987ecaf91`, file SHA-256 `5eddd42914ab74426257ab8c1ff883b25c0c39aaa40bf7c40ccc2e363fbb3957`, pending independent Pass 1 review. P3 Pass 2 must not begin until the accepted P2 graph/observation/route interface and exact review are entered in its ledger.
- **Accepted Plan A owner response:** `03b7a9f5cbb161a2b30563e76a360619df144849`, SHA-256 `e6863a2b3b52fc2526062ac081cec0b153742d0f32344d8006d766cfad74df21`; exact Pass 3 ACCEPT record SHA-256 `9a1eea2c57c1c20b59d6fdfd11e7ffb697aa77b0093f7fcb40a44978233f38e2`.
- **Accepted reconciliation/profile:** `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; canonical profile hash `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`; V4 proposal/profile digest `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.

## Review scope and evidence

The reviewed plan's exact Plan A contract matches the accepted overall plan and Plan A owner response: 12,096 padded nodes gated by `node_valid`; 96,768 node-major ordered support rows at eight descriptors per padded node; ten directed edge slots; 14 ancestor levels; adopted float64 storage for distance, route cost, and parent-edge length; and the fixed utility, behavior-logit, mask, and identity interfaces. It preserves the distinction between planner support and native execution/safety, does not introduce Plan C's K cap into Plan A, and leaves PPO sampling and sampled-action/old-log-probability linkage to P5.

The accepted source base supports the plan's cited implementation surface. `frc_defense/meco_tracks.py` defines A1/A2/A3 at lines 113–142, 204–238, and 241 onward. `frc_defense/meco_plan_a_policy.py` contains the support structure and digest inputs at lines 28–47 and 80–146, support enumeration/order at lines 359–378, and ancestor-jump construction at lines 393–399. The same base uses dynamic scalar extraction and host transfers in support construction/digest (`meco_plan_a_policy.py` lines 112–146, 283–324) and CPU serialization in the digest at lines 104–112. The plan treats these as existing behavior requiring future GPU implementation evidence; it does not claim current residency or runtime conformance.

The allowlist is finite and appropriate: the Plan A policy adapter, a standalone typed Plan A V4 adapter, and two scoped test files. The plan excludes `meco_track_training.py` and assigns it exclusively to P5; it excludes shared `meco_tracks.py`, P2 graph/environment/route producers, P1 schema/artifact modules, and P6 qualification paths. The P1 package plan is pinned and accepted, while the P1 implementation/API and P2 accepted API remain explicit pre-Pass-2 gates. The P2 candidate is not treated as accepted. The absence of a pinned P2 accepted head is a defined wait condition, not an assumption that an interface exists.

Validation gates are concrete and consistent with the accepted overall plan and repository AGENTS.md: exact capacity/order/mask fixtures, device-only algorithmic validation on both assigned WX9100 devices with UUID/runtime and residency evidence, seeded end-to-end simulator evaluations for A1/A2/A3 with inspected records/replays, and a fresh replay only after restarting and verifying the current dashboard backend/frontend and endpoints. G0 protects the active campaign and requires user manual stop, independently confirmed lease release, and separate authorization. The plan does not claim those gates have been met. Planner, implementer, and reviewer roles are distinct, and Pass 2/3 are blocked on exact dependency heads and assigned identities.

## Review limits

This ACCEPT applies only to the exact P3 package plan commit and file hash above. It accepts the plan for its bounded future P3 scope; it is not implementation approval, human maintainer approval, ABI publication, training authorization, qualification, or evidence that any runtime/device/dashboard/campaign gate is complete. No tests, source edits, runtime, simulator, GPU, dashboard, or campaign actions were performed in this review.
