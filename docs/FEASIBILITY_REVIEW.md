# Feasibility review of the 10% reduction plan

Reviewed 2026-10-06 against the current worktree. Implementation files were not edited. The original 35,196-line baseline and <=3,519 target remain fixed. The worktree has since changed: eight inventory hashes differ and an additional focused-playback test exists; the verification snapshot counts 35,310 lines. Those concurrent changes are not cleanup savings.

## Verdict

**The proposed 90% reduction is not supported by evidence under the full preserved scope. It is infeasible as deletion-only cleanup. Its rewrite feasibility remains unproven.** The prior 12–18k first-stage estimate and 3,519-line allocation were guesses; they are not validated implementation budgets. The plan has been corrected accordingly.

Smaller source is achievable. Seventeen generated files totaling **2,489 lines (7.1% of baseline)** were recreated byte-for-byte from current canonical sources using the installed PyTorch hipify tool in an isolated directory on the HDD pool. This verifies source reproducibility, not native compilation. Removing these checked-in/generated source copies from the maintained source is a credible first cut after build-output placement is fixed. Four unreferenced hipified-looking files were not regenerated from current canonical sources and should not be included in that verified saving.

## Budget verification

Runtime Python loaders name **29 canonical C++/CUDA build inputs totaling 5,217 lines**, 148% of the entire final target. This is the union of existing required/optional runtime build paths, not a dynamically profiled claim that every kernel executes on every tick. Some could be retired after profiling and parity proof. Preserving all these implementations unchanged already prevents the 10% result.

Baseline disjoint size buckets:

| Bucket | Lines |
|---|---:|
| Python package runtime | 14,069 |
| Canonical runtime build inputs | 5,217 |
| Other native/generated sources | 3,229 |
| Browser frontend | 3,485 |
| Scripts including untracked audit | 6,559 |
| Tests | 2,637 |
| Total | 35,196 |

Even the unrealistic deletion of **all** scripts, tests and other native sources leaves **22,771 lines (64.7%)**. That calculation removes required launch/performance tools and safety checks; it is an optimistic stress calculation, not a recommended action or an irreducible theoretical lower bound. Production code still needs a further 84.5% reduction from that remainder to reach 3,519.

Measured current components versus the speculative plan budgets:

| Component | Current measured lines | Previous target | Required shrink |
|---|---:|---:|---:|
| Core physics/planner/field/observations/two-robot and 3v3 game code | 6,836 | 950 | 86.1% |
| Canonical native sources | 5,217 | 1,200 including loader | >77.0% |
| Training family including model, PPO, DDP, generation and evaluation | 3,060 | 600 | 80.4% |
| Replay backend and baseline frontend | 4,795 | 450 | 90.6% |
| Baseline scripts/tests | 9,196 | 319 | 96.5% |

These are illustrative group comparisons, not a disjoint total: some wrappers, CPU planner and package support are additional, and current measurements differ slightly from the baseline. No prototype demonstrates any of the proposed target component sizes. A 450-line replay budget is especially unsupported when it includes seeded six-robot GPU generation, durable jobs, historical recordings, rich route/fuel inspection and browser controls.

## Architectural constraints checked

1. **Shared physics is already implemented.** `tensor_physics.py` accepts `num_robots`, stores `[world, robot, ...]` tensors, and is used by both environments. The plan cannot count adopting this layout as new savings. Remaining gamepiece/strategy/observation duplication must be evaluated individually.
2. **Equal NN dimensions do not imply equal semantics.** Both strategic environments use 137 inputs and eight categorical actions, but the 3v3 observation uses teammate columns. `tensor_training_3v3.py` zeroes input weights at columns 58:78 when transferring a non-3v3 checkpoint. A universal observation rewrite needs explicit semantic versions and adapters; simple same-shape loading would change policy behavior.
3. **Robot-count fast paths differ.** Physics selects separate two-robot contact-pipeline and six-robot multi-collision paths. Generalizing the Python code cannot justify deleting a faster specialized kernel without measurements.
4. **Several proto modules are production dependencies.** Current environment imports and calls the full/candidate strategic observation and perception-commit implementations. Their generated bindings are reproducible; their canonical implementations are not disposable by name.
5. **CPU AD* has a live playback compatibility use.** The dashboard reconstructs historical paths with `ADStarPlanner`. Recording future routes does not make old recordings compatible; test a migration or retain reconstruction.
6. **Checkpoint resume needs a more precise contract.** Current saved payloads and initial loading provide policy weight warm starts, not proof of exact optimizer/RNG/rollout continuation. The reduced implementation should preserve existing service restart behavior. Exact interrupted-training continuation would be added scope, not an established feature to assume.
7. **Loader consolidation has concurrency constraints.** Current loaders mutate PyTorch build globals and use shared space-safe aliases. Centralization must serialize initialization/build mutation and preserve per-kernel compiler flags; generation evaluation can use multiple worker streams.

## Verification performed

- Queried the existing graph, then inspected current source because graph source locations are stale.
- Checked literal native input references in every top-level runtime Python module, counted sources, and checked inventory SHA-256 drift.
- Ran hipify on copies of all 29 canonical native inputs under `/home/brian/Projects/autodrive-feasibility-d0z4_ch9`; 17 existing generated outputs match byte-for-byte. No production source/cache was changed and no kernels were compiled.
- Confirmed PyTorch `2.14.0+rocm10.2.0a20260927`, HIP `7.17.26385`, and two available AMD Radeon Pro WX 9100 devices.
- Ran `tests/test_focused_playback_setup.py`, `tests/test_strategic_held_interval.py`, and `tests/test_tensor_training_eval_dispatch.py`: **11 passed in 2.88s**. These cover shared staging/masked reset, focused playback sizing, held-action terminal boundaries/GAE, and evaluation dispatch. They establish baseline contracts, not rewrite equivalence or GPU performance.

No comparative GPU throughput run was performed: no reduced implementation exists to compare. No clean native build, DDP training run, full-match parity run or browser integration test was performed. GPU availability alone does not establish optimized execution.

## Revised feasible approach

- Start with the verified 2,489-line generated-output opportunity. Preserve canonical sources, configure isolated build outputs, and confirm clean native compilation before removing copies.
- Inventory redundant tooling at function level and retain unique parity/performance procedures; do not assume all 6,559 script lines can go.
- Simplify the inspection UI and remove legacy-global training plumbing in separate, measurable changes. Recount actual savings instead of assigning unsupported percentages.
- Prototype one genuinely duplicated subsystem, such as candidate selection or gamepiece updates. Keep separate checkpoint semantic adapters and specialized kernel paths. Require fixed-seed parity and equal-or-better end-to-end GPU performance before extending the consolidation.
- Rebudget using the smallest validated implementation after that prototype. Do not undertake the full shared-core rewrite on the strength of the current 3,519-line table. If the measured preserved implementation remains above 10%, report the achieved reduction and remaining scope tradeoffs explicitly.

The next verification milestone is a clean-build generated-source removal and one shared-subsystem prototype with actual LOC/performance deltas. That is the evidence needed to distinguish an ambitious but achievable rewrite from a target that requires dropping features.
