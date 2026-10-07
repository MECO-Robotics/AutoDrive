# AutoDrive codebase reduction plan

Inventory date: 2026-10-06. Base commit: `900cf0571e65330b38a350f83086507d3d5f4ccc`.
Scope: current working-tree source, including existing modifications and untracked source. Inventory by three GPT-6 Luna subagents (playback, GPU training/generation, deterministic/NN defense), with parent verification. No implementation files were changed for this assessment.

## Decision

Target at most **3,519 physical source lines**, 10% of the current **35,196**. Count Python, C++/CUDA/HIP, JavaScript, CSS, HTML, shell, tests and tooling. Exclude environments, checkpoints, evaluation recordings, Git history, generated analysis reports, and this planning documentation. Keep the denominator fixed; do not claim success by moving code outside the count, replacing it with opaque binaries, or minifying it.

**This is a rewrite target, not an achievable deletion-only cleanup.** Current evidence does not establish that every preserved behavior and the optimized GPU path can fit in 3,519 lines. The earlier 12–18k first-stage estimate is withdrawn as unsupported. A deletion-only stress calculation still leaves 22,771 baseline lines even if every script, test and noncanonical kernel source is removed; that hypothetical would also discard necessary tooling and checks. A second, shared-core rewrite must prove whether 10% is feasible. If its validated minimum exceeds the budget, report the result rather than silently removing behavior.

**Feasibility review:** see `FEASIBILITY_REVIEW.md` and `feasibility-evidence.json`. Canonical runtime GPU build inputs total 5,217 lines; 17 generated copies totaling 2,489 lines were reproduced byte-for-byte. The 10% target is unsupported under the full current scope; deletion-only feasibility is ruled out. A rewrite is not proved impossible, but must earn its budget through a prototype.

Interpret preserved scope conservatively: retain recorded inspection and current seeded GPU scenario generation, deterministic and NN defense, deployed two-robot generational training, and the six-robot scenario behavior used by playback. “Generation” covers PPO generations and playback generation. Do not assume 3v3 can be discarded merely because the service trains a two-robot environment.

## Measured baseline

| Area | Tracked files | Physical lines |
|---|---:|---:|
| Package, kernels and browser assets | 100 | 25,171 |
| Scripts | 41 | 6,475 |
| Tests | 21 | 2,637 |
| Other tracked docs/config/deploy | 6 | 144 |
| All tracked files | 168 | 34,427 |
| Tracked source only | 161 | 34,257 |
| Untracked source: two HIP outputs and playback audit | 3 | 939 |
| Current source denominator | 164 | 35,196 |

The accompanying `codebase-inventory.csv` records each source path, tracking status, line/byte counts and SHA-256. These are physical newline counts, not logical statements or nonblank LOC.

The checkout occupies approximately 11 GiB, dominated by `.venv` (~6.78 GB), `evaluations` (~3.49 GB), `checkpoints` (~0.78 GB), and `metrics` (~0.05 GB), using decimal byte estimates. Source is only a few MB. Code reduction and disk reduction are separate projects. Do not delete checkpoints or evaluation evidence as code cleanup. Archive project artifacts on the All Drives pool; new worktrees default to `/home/brian/Projects`. The current checkout already resolves onto that pool. Environment placement is not an intended change in this plan.

## Feature and dependency inventory

| Preserved feature | Current implementation | Required contracts |
|---|---|---|
| GPU PPO and generations | `tensor_training.py`, `tensor_training_runtime.py`, `tensor_training_ppo.py`, `tensor_training_generational.py`, evaluation/reporting modules | Device-resident rollouts and updates, seeded evaluations, strong/diverse population, atomic checkpoints, compatible resume, multi-GPU DDP |
| Deterministic/NN defense | `tensor_sim.py`, `tensor_defense_{opponent,observation,reset,step,gamepieces}.py`, `tensor_adstar.py`, ActorCritic | Same simulation and observation constraints; deterministic strategy/controller and NN selection; no hidden attacker goal; correct action masks, latency and normalization |
| Physics and game state | `tensor_physics.py`, `tensor_physics_collision.py`, `tensor_gamepieces.py`, `field.py`, supporting wrappers | Swerve/current/friction constraints, contacts, active masks, front intake, possession, hub activity and scores |
| Six-robot playback scenarios | `tensor_3v3.py`, `_tensor_3v3_*.py`, `dashboard_simulation.py` | Six robot control selections, seeded generation, fuel owners/positions, routes, targets, match clock and score history |
| Inspection UI | `dashboard.py`, `dashboard_api.py`, `dashboard_static/*` | Stored training/evaluation/generation runs, selection, play/pause, scrub, speed, field/fuel/path rendering, state inspection |
| Optimized AMD GPU execution | Runtime kernel wrappers and canonical `csrc/*.cpp`/`*.cu` sources | Fused swerve/contacts/perception/gamepieces/ranking/planning; graph-compatible paths; correct compiler flags and build discovery |
| Operational launch | `scripts/run_gamepiece_training.sh`, systemd unit, ROCm requirements, project entry points | Existing launch, resume, output paths and status/error reporting |

Production trainer path: CLI → generational orchestration → PPO → TensorDefenseEnv → physics/gamepieces/planner/observation → fused kernels or Torch fallback.

Playback generation path: dashboard scenario job → TensorThreeVsThreeEnv → shared tensor simulator/physics/planner → batched snapshots → JSON → browser field renderer.

Important boundaries:

- The two environments share physics and the model class, but have distinct observation/action/game strategy implementations. Consolidating them is a redesign, not file concatenation.
- All 43 current top-level package Python modules are reachable by a conservative static relative-import traversal rooted at package initialization, training and dashboard. Reachability includes conditional and function-local imports; it does not prove that every branch is needed, but rules out blanket “unused module” deletion.
- `dashboard_simulation.py:592` reconstructs legacy playback routes using the CPU `ADStarPlanner` from `adstar.py`. Replace that contract before deleting the 1,085-line CPU planner.
- `tensor_strategic_observation_proto.py`, its full variant, and `tensor_perception_commit_proto.py` are imported by the active environment. “proto” does not mean dead.
- Most fused extension loaders are ROCm-specific. NVIDIA CUDA currently has the Torch GPU path; equivalent custom-kernel optimization on NVIDIA is not verified. Preserve existing hardware behavior and benchmark on the actual target hardware.
- `scripts/generate_nn_vs_nn_playback.py` uses a direct-command simulator rather than the full strategic fuel game. Treat as a candidate legacy tool, not proof that strategic NN-vs-NN gameplay already exists.
- The existing graph has stale source locations (for example, the old monolithic trainer). It was queried for orientation; findings above were checked against current files.

## Concrete cut candidates

| Candidate | Evidence / size | Planned action and dependency gate |
|---|---|---|
| One-off scripts | 6,475 tracked lines, mostly benchmark/prototype/test investigations | Retain one E2E performance harness, selected kernel parity gates, one replay audit and operational launch/campaign commands. Absorb unique validation before removing redundant scripts. No exact savings assumed yet. |
| hipify output | 1,446 tracked lines in `*_hip_hip.cpp` and `.hip`; another 855 untracked HIP lines | Canonical loaders point at `.cpp`/`.cu`. Rebuild from canonical sources in an isolated cache, then remove generated source copies from Git and generate outside the source tree. Do not touch active JIT outputs. |
| Exact duplicated kernels | Contact pipeline canonical/proto `.cu` pair: 414 lines each; occupancy hipified/proto binding pair: 54 lines each | Keep canonical implementation; remove prototype copy only after its experiment is retired. Occupancy pair overlaps the generated-output candidate; do not double-count. |
| Repeated extension setup | Similar ROCm SDK/space-safe alias/compiler/JIT setup across wrappers | One small extension loader and an explicit kernel manifest. Keep per-kernel flags and ABI checks; avoid new generic framework abstractions. |
| Training facades / compatibility plumbing | ~3k lines across training modules, late binding and legacy globals | Direct explicit trainer/evaluator calls; one model/checkpoint loader; one snapshot writer. Preserve checkpoint compatibility and DDP reducer/control-group behavior. |
| Dashboard presentation extras | Main CSS 1,119, JS 980, field renderer 631, HTML 400, field guide 262 lines | Rebuild around inspection: timeline + field + state panel + run selector. Drop tutorial, decoration and redundant charts first. Keep route/fuel/control inspection and generation jobs. |
| CPU planner route reconstruction | 1,085-line `adstar.py`, used by legacy playback completion | Record routes from the live planner; add an offline migration for historical recordings; remove CPU reconstruction only after fixture equivalence. |
| Parallel backend/reference branches | Torch references and prototype kernels coexist with production fused paths | Keep validation references until parity is demonstrated. Retire superseded implementations one at a time; retain small independent oracle fixtures or a justified fallback. |
| Separate 3v3 code | ~2.8k environment/mixin/trainer lines, with active playback dependency | Retain behavior. Share state/physics/planning/policy/snapshot contracts through a common robot-axis layout. Do not delete the feature as a shortcut. |
| Existing tests | 2,637 lines | Consolidate repetitive fixtures and parameterize backend/mode parity. Retain feature contracts, physical constraints and checkpoint/resume gates. |

## Execution sequence

1. **Freeze evidence and contracts.** Create an isolated worktree on the HDD pool, preserving all current working-tree changes as the baseline. Store representative existing checkpoints and training/evaluation/3v3 playback fixtures. Inventory CLI/API/checkpoint schemas. Record fixed-seed states, fuel/score events, routes, rewards and terminal flags. Establish end-to-end baseline timing on the target GPU; no production service interruption.
2. **Remove reproducible duplication.** Prove clean JIT builds without committed hipify outputs; exclude generated copies. Retire exact duplicate experimental kernels. Consolidate benchmark scripts only after migrating unique checks. Recount lines after each change. This phase alone cannot reach 10%.
3. **Collapse plumbing.** Share extension loading, checkpoint handling, snapshot schemas and training/evaluation interfaces. Remove legacy globals dispatch and repeated payload assembly. Keep atomic writes, resume checks, DDP coordination and efficient snapshot transfers.
4. **Replace the dashboard with an inspection viewer.** Use one field renderer, a minimal API, compact styles and a versioned recording schema. Preserve controls, fuel ownership, route overlays and seeded deterministic/NN scenario generation. Migrate legacy route reconstruction before retiring CPU AD*.
5. **Prototype shared game semantics before committing to a rewrite.** Physics already represents robot count as a tensor axis and is already shared; do not count that as new savings. Attempt sharing gamepiece rules and candidate selection first, preserving separate observation/action adapters and robot-count-specific GPU fast paths. Deterministic and NN policies consume the same public state boundary. Add thin two-robot and six-robot adapters to preserve historical checkpoint observations/actions. Replace the old cores only after parity and throughput gates pass. This is the largest and riskiest step.
6. **Attempt the 10% implementation.** Start with the measured shared core; implement only the preserved interfaces. Deduplicate kernel helpers and bindings without losing launch fusion or determinism. Measure code size and performance together. If the real minimum cannot fit 3,519 lines, publish the achieved size and remaining irreducible code; keep the validated implementation instead of breaking features to satisfy the number.

## Unvalidated 10% architecture budget

This is an aspirational design budget, **not a validated estimate**. Existing canonical GPU sources alone exceed the entire final budget, so this requires substantial kernel consolidation and a successful shared-core redesign.

| Responsibility | Final physical-line ceiling |
|---|---:|
| Tensor state, physics/gamepieces, planner, observations, deterministic policy | 950 |
| Canonical fused kernels and bindings, shared extension loader | 1,200 |
| NN model, PPO/DDP, generations/evaluation, checkpoints and CLI | 600 |
| Replay capture/generation API and inspection browser | 450 |
| Essential parity/performance tests, launch tooling | 319 |
| Total | 3,519 |

Prefer a few cohesive modules (`game`, `planner`, `policies`, `train`, `replay`, `extensions`) and canonical kernel sources. File count is secondary to readable, reproducible source size. Do not flatten code solely to reduce physical lines. A smaller file count with the same duplicated logic is not success.

## Acceptance gates

- **Behavior:** full 160-second matches, deterministic and NN control in both existing scenario sizes; front intake, fuel possession, score/hub schedule, collision/traction constraints, route legality and visibility restrictions. Fixed-seed comparisons cover terminal and partially active batches, observation history/RNG and held-action boundaries.
- **Learning correctness:** GAE termination/truncation semantics, action masks and log probabilities, finite updates, compatible checkpoint roundtrip/resume and explicit mismatch rejection; PPO generation produces evaluated strong/diverse population and inspectable records.
- **GPU performance:** warmed matched A/B/A runs with identical seeds, environment count, horizon, physics/control cadence, dtype, device, recording settings and compiler options. Measure full training iteration, world-steps/s, generation/evaluation time, scenario-generation time and peak VRAM. Verify required fused kernels execute and absence of new per-tick host synchronization. Preserve DDP behavior. Proposed gate: no >5% median E2E regression outside measured noise, and no OOM at baseline batch size. This tolerance is a plan proposal, not a user-approved degradation allowance; seek parity or improvement.
- **Replay:** load historical training/evaluation/generation fixtures; inspect six robots, fuel movement/ownership, score transitions, match time, routes and controller targets; play/pause/scrub/speed/run selection work. Seeded graph/eager snapshot comparison and generation job completion/reattach work.
- **Reproducibility:** isolated clean JIT cache builds from retained canonical source; required licenses stay; launch entry points and systemd resume remain functional. Keep a rollback checkpoint between phases.
- **Size:** use the CSV extension set and fixed baseline. Count all handwritten source, essential tests and tooling; report generated material separately. The final source must be <=3,519 lines to claim 10%.

Original inventory ran no behavioral or GPU benchmarks. The follow-up feasibility check ran 11 focused existing tests successfully and reproduced generated sources; it did not run comparative training throughput, clean native compilation or full-match parity. “Fully optimized” remains a performance requirement to verify against the measured current path.
