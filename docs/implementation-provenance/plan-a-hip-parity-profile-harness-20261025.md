# Plan A HIP parity/profile harness implementation candidate

**Plan:** `docs/plans/plan-a-hip-parity-profile-harness-20261025.md`
**Plan SHA-256:** `0880e0880899cd737a9ac8f6c4fb14320f75f7abf24164cb53e3f9fb0e7a5505`
**Planner/reviewer ledger head:** `5e9667b7a99bf7dc2ab8feea57de6e0e903422cf`
**Implementation branch:** `codex/plan-a-hip-parity-profile-harness-impl-20261025`
**Implementation candidate:** `ca4604e019826cb6b2a010f84872d0eb15f83d16`
**Independent implementation review:** approved at exact head `ca4604e019826cb6b2a010f84872d0eb15f83d16`

## Implemented scope

- `scripts/verify_plan_a_hip_parity.py` defaults to immutable `offline_fixture_only` reports. It records no HIP import, compilation, or device discovery and does not edit training configuration.
- `tests/fixtures/plan_a_hip/manifest.json` binds canonical manifest content, fixture graph/tensor hashes, expected per-output CPU hashes, support digests, and exact hashes for the maintained oracle, pose graph, support digest, fixture recipe, native binding, HIP kernel, extension build loader, harness, and CLI.
- Fixtures cover equal-distance node ordering and first-slot parent retention, the strict `1e-9` relaxation boundary, unreachable nodes, graph geometry with a nonzero root connector, active invalid edge and geometry data, and a deterministic 64,000-node × 10-slot 8-heading pose-lattice memory/capacity case.
- Explicit `--mode device --device cuda:{0,1}` candidate mode first acquires the repository's exclusive `lease_devices` lock for the selected indexed accelerator, then runs `verify_post_lease_wx9100` while holding that lock. It verifies that PyTorch identity matches the physical attestation before importing the on-demand extension. A busy device lock fails closed, so an active training lease prevents the candidate run. The report records the generated lease ID and physical attestation; it does not trust a caller-supplied lease ID or `training_leases_released` flag. The candidate exercises invalid-input rejection, exact five-field route parity, route-derived `plan_a_support_digest`, both native paths for geometry cases, synchronized route timing distributions, separate compile/first-use costs, and peak allocated/reserved memory. It cannot open the production gate or change normal support dispatch.
- Reports are created with exclusive file creation in a unique timestamp/UUID path. Failed invocations also write a failure record.

The device candidate requires the selected repository lock to be available, acquires it exclusively for the whole candidate run, and requires the post-lease physical identity attestation to match the selected WX9100. A clean checkout and exact pinned source/fixture identity are checked at entry and revalidated before a passing report is returned. No device invocation or lease inspection/acquisition occurred in this implementation pass; both WX9100 leases remain subject to orchestrator scheduling after continuous training releases them.

## Static checks and deferred evidence

Python AST parsing passed for the harness, fixture recipe and CLI. `git diff --check` passed. No tests were added or run under the active instruction. Fixture authoring executed the CPU oracle once per valid fixture to freeze expected outputs; this was fixture generation, not a test run. The source/lease follow-up checkpoints were AST-parsed and diff-checked only.

The offline CLI itself was not run. No HIP extension import/compile/probe, device/lease inspection or acquisition, WX9100 parity/profile, simulator, training, replay, dashboard, or deployment work was performed. Independent implementation review approved exact head `ca4604e019826cb6b2a010f84872d0eb15f83d16`; reviewer confirmed the actual exclusive flock, post-lease physical attestation before extension loading, fail-closed source/fixture/checkout revalidation, unchanged closed production gate, and no hardware/test actions. Hardware verification remains deferred until the orchestrator confirms continuous training has released the exclusive leases and schedules the run. The production gate remains false, and normal training/support dispatch remains on the CPU oracle.
