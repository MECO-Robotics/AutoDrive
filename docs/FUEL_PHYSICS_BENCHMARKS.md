# Fuel physics benchmark protocol

The fuel model uses individual 3D compliant spheres, while robot chassis dynamics remain planar. It includes fuel contacts, friction, rolling resistance, gravity, static field contacts, and reciprocal robot bumper impulses. Coefficients are provisional calibration parameters; wheel climbing and foam deformation geometry are not represented.

## Reproducible comparisons

Run `scripts/benchmark_fuel_physics.py` using the project `.venv`. Seeded initial conditions are generated on CPU and copied to each device. Each timed repetition constructs fresh state after warming disposable state, and synchronizes its own device before and after timing. Initialization, JIT compilation, host quality diagnostics, garbage collection of discarded warmup state, and state copies are outside the timed interval. Contact and neighbor buffer byte counts are reported directly; global allocator peaks can include graph reference state and should not be treated as one simulator's footprint.

The default workload is 24 worlds with 504 balls and six robots, a 2 m/s robot forward command, 64 control ticks (1.28 simulated seconds), three repetitions, and either four 5 ms or ten 2 ms physics substeps per 20 ms control tick. The robot drivetrain and fuel advance together at each substep. Timing excludes policy inference, observation generation, and planning; the optional full environment comparison measures a separate scope.

Scenes:

- `sparse`: jittered separated balls with seeded planar velocities.
- `pile`: a tangent 9 × 8 × 7 stack of 504 spheres that settles while a robot drives toward it.
- `settled`: separated stationary balls on the floor, long enough to exercise sleeping.

Variants:

| Variant | Purpose |
| --- | --- |
| `robot_only` | Identical robot substeps without fuel contact updates |
| `allpairs` | All fuel pairs; matched solver and timestep reference |
| `grid` | Spatial candidates rebuilt every substep |
| `cached` | Conservative candidates reused until motion exceeds skin bound |
| `compact` | Compact actual contacts before their impulse calculation |
| `colored` | Native complete-graph round-robin coloring; independent pairs use sequential color passes |
| `colored_torch` | Torch reference using greedy contact graph coloring |
| `sleep` | Cached grid with sleeping enabled |
| `graph` | Actual graph capture/replay with fixed substeps; rejects capture failures or short eager trajectory divergence |
| `skin025`, `skin15`, `skin30` | Change neighbor margin to 0.025, 0.15, or 0.30 m; cached baseline uses 0.075 m |
| `neighbors16`, `neighbors32` | Reduce neighbor capacity; preserve overflow fallback |
| `island_sleep` | Sleep and wake connected contact groups |
| `robot_allpairs` | Cached fuel grid with robot–fuel all-pairs checks |
| `compact_full` | Compact actual contacts using full all-pairs buffer capacity |

Quality outputs include finite state, maximum ball and floor penetration, robot forward speed/displacement, fuel planar speed, sleeping fraction, repeated-run state difference, and state difference from a matched all-pairs run. Reported aggregate momentum includes external drivetrain and floor forces; it is not a conservation assertion. Isolated momentum conservation is covered by physics correctness tests.

Fixed-step runs isolate algorithm cost. `--adaptive` instead computes a conservative motion bound once per control tick and reports actual substep counts. Graph experiments deliberately use fixed substeps; their validation covers four early ticks below a 1e-4 maximum state difference, and separately reports long trajectory difference. Dense pile trajectories amplify atomic summation differences, so long results require comparing physical invariants and aggregate outcomes as well as individual states.

Runtime wins are meaningful only alongside acceptable penetration, short trajectory parity, and contact-buffer overflow handling. Contact stiffness can require more substeps; selecting the fastest configuration alone does not establish physical accuracy.

## Dual GPU execution

On 2026-10-08 the host exposed two AMD Radeon Pro WX 9100 cards (`gfx900`), each initially idle, with no KFD processes. PyTorch reported `2.14.0+rocm10.2.0a20260927` and HIP `7.17.26385`.

Raw reports and logs are stored on the HDD-backed project filesystem under `outputs/fuel_physics/2026-10-08/`. Extension caches are separate per process/device on that filesystem. `/tmp/autodrive-fuel-extensions-gpu0` and `gpu1` are symlinks to those caches so build tools receive paths without spaces. Existing cache modules were copied rather than modified in place.

Example commands (execute simultaneously in separate shells):

```bash
TORCH_EXTENSIONS_DIR=/tmp/autodrive-fuel-extensions-gpu0 MAX_JOBS=2 \
  .venv/bin/python scripts/benchmark_fuel_physics.py \
  --device cuda:0 --variants robot_only allpairs grid cached \
  --output outputs/fuel_physics/2026-10-08/gpu0_matrix.json

TORCH_EXTENSIONS_DIR=/tmp/autodrive-fuel-extensions-gpu1 MAX_JOBS=2 \
  .venv/bin/python scripts/benchmark_fuel_physics.py \
  --device cuda:1 --variants robot_only allpairs compact sleep \
  --output outputs/fuel_physics/2026-10-08/gpu1_matrix.json
```

Swap representative variants across GPUs afterwards to detect device bias. CPU dispatch and host contention remain shared resources during simultaneous runs; cross-device runs are complementary throughput experiments, not independent isolated machine measurements.

## Initial existing-physics baseline

With native HIP robot optimizations explicitly disabled, the preexisting six-robot Torch physics measured:

| Device | Worlds | Median per 20 ms tick | Repetitions |
| --- | ---: | ---: | ---: |
| WX 9100 GPU 0 | 24 | 18.04 ms | 3 × 32 ticks, after 8 warmup ticks |
| WX 9100 GPU 1 | 24 | 19.11 ms | 3 × 32 ticks, after 8 warmup ticks |

This is a Torch fallback robot-physics baseline, not a full environment or optimized training baseline. Do not use it to assert overall training slowdown. Raw reports: `baseline_torch_gpu0.json` and `baseline_torch_gpu1.json`.

## Results

The first corrected contact solver comparison completed on both GPUs with stable source hashes and no benchmark errors. These results precede the follow-on robot query, bounded compact buffer, terrain, and island stages. Reports ending in `_provisional` are earlier diagnostics that must not be used to select final settings.

GPU 1, 24 worlds × 504 balls, 64 ticks × 3 repeats, 2 m/s robot command:

| Solver path | Sparse, 5 ms | Sparse, 2 ms | Pile, 5 ms | Pile, 2 ms |
| --- | ---: | ---: | ---: | ---: |
| All pairs, Jacobi | 3.340 ms | 8.297 ms | 3.867 ms | 9.656 ms |
| Grid rebuilt, Jacobi | 3.197 ms | 7.989 ms | 3.367 ms | 8.322 ms |
| Cached grid, Jacobi | 2.936 ms | 6.690 ms | 3.186 ms | 7.600 ms |
| Compact contacts, Jacobi | 3.161 ms | 7.249 ms | 3.189 ms | 7.627 ms |
| Native colored | 7.090 ms | 17.026 ms | 9.534 ms | 23.359 ms |

Times are per 20 ms control tick for coupled robot/fuel physics. For the dense pile, cached grid at 2 ms is 1.27× faster than matched all-pairs. These are measured algorithm ratios, not total training ratios.

The timestep comparison exposed a quality problem: dense pile maximum overlap was approximately 6.7 cm at 5 ms, versus 1.7 cm at 2 ms. Fuel energy at the end was approximately 388 J versus 288 J; robot travel was 1.64 m versus 1.48 m (2.30 m without fuel). Thus the nominally faster 5 ms setting is not an adequate basis for selecting realistic production defaults.

GPU 0 repeated representative paths without the earlier unrelated dashboard job. At 2 ms in the pile, cached grid measured 7.297 ms and fixed-cadence graph replay measured 6.242 ms (14.5% reduction). All four graph scenes/timestep combinations passed short eager parity below 1e-4; long pile differences were comparable to eager repeated-run variation. Graph throughput is a prototype result and excludes production adaptive cadence and environment pointer rebinding.

Raw corrected phase reports: `gpu1_matrix.json` and `gpu0_crosscheck.json`. Source hashes are embedded in each report. No published speedup or earlier theoretical estimate is used as a measured result.

Final selected-source measurements are recorded below.

## Selected settings and optimization inventory

Production uses ten nominal 2 ms substeps per 20 ms control tick with adaptive motion bounds, cached fuel spatial neighbors, Jacobi compliant contacts, 0.075 m neighbor margin, and 64 neighbor slots. Six robots scan fuel directly: repeated 5 × 64 tick comparisons on both GPUs showed robot queries were faster with all pairs than with a spatial mask at this workload.

| Device | Sparse robot grid | Sparse robot all pairs | Pile robot grid | Pile robot all pairs |
| --- | ---: | ---: | ---: | ---: |
| GPU 0 | 7.508 ms | 6.695 ms | 8.903 ms | 7.589 ms |
| GPU 1 | 7.519 ms | 6.710 ms | 8.937 ms | 7.603 ms |

These times include the same cached fuel grid and ten physics substeps. Variant order was reversed across GPUs. Robot spatial queries remain available for different workloads.

| Opportunity | Implementation and selection |
| --- | --- |
| Individual moving fuel | 3D spheres, gravity, spin, compliant frictional contacts and reciprocal robot impulses; selected |
| Spatial broadphase | Native linked cells and safe cached neighbor lists for fuel; selected |
| Conservative cache invalidation | Motion/ownership/activity changes rebuild candidates; selected |
| Contact capacity | Overflow uses safe all-pairs fallback; selected |
| Robot spatial queries | Implemented and parity tested; direct scans selected for six robots and 504 balls |
| Contact compaction | Implemented; bounded grid capacity versus full capacity tested; fused path remains default |
| Parallel solver | Jacobi selected; native colored sequential color passes implemented and tested, slower in these workloads |
| Sleeping | Per-piece sleeping selected; connected island sleeping implemented and tested, optional because no throughput improvement at 24 worlds |
| Physics cadence | 5 ms and 2 ms tested; 2 ms selected for pile quality, with adaptive bounds |
| GPU graphs | Actual capture/replay and short parity tested; benchmark prototype, not enabled in production environment |
| GPU memory/dispatch | Device-resident broadphase/solve, fused contacts, occupied-cell clearing, contiguous buffers; selected |
| Pile aggregation | Excluded because it changes individual fuel dynamics required by this simulator |
| Warm-started constraint iterations | Not a separate mode: the chosen compliant per-substep impulse model does not have accumulated hard-constraint multipliers to warm start |

The compact contact buffer at 24 × 504 × 64 capacity uses **6,193,152 bytes**, versus **24,337,152 bytes** for full pair capacity, a **74.6% reduction**. Overflow fallback preserves contacts. Neither compaction nor coloring automatically implies faster whole-simulator execution.

Short one-variable tuning tested margins 0.025/0.075/0.15/0.30 m and capacities 16/32/64. Sparse and pile timings favored different margins, and the short run did not establish a repeatable improvement sufficient to change the conservative 0.075 m/64 defaults. All those short comparisons preserved contact outcomes without observed overflow; separate correctness tests deliberately exercised overflow fallback.

## Correctness verification

Final selected configuration:

```bash
.venv/bin/pytest -q tests/test_fuel_physics.py tests/test_fuel_physics_hip.py \
  tests/test_fuel_physics_integration.py tests/test_fuel_physics_sleep.py
```

The combined suite passed **66 cases in 76.71 seconds**. After a final two-line masked-reset snapshot fix, **seven targeted reset cases passed in 4.14 seconds**, including three new cases on CPU, GPU 0, and GPU 1. This validates **69 unique cases**; it is not a single 69-case run.

Cases cover linear/angular momentum, inactive masks, pending external spawn/ownership changes, sleeping/waking, terrain support and spawn height, robot spatial queries, contact capacity overflow, colored solving, and environment integration. Earlier 60-case results are superseded by this final validation.

Selected existing suites `tests/test_tensor_env_active_mask.py` and `tests/test_tensor_3v3.py` produced **13 passes and two failures**, both reproduced with `fuel_physics=False`: `TensorEnvActiveMaskTests::test_inactive_world_state_planner_and_rng_are_untouched` (tactical planner `last_progress`) and `test_defender_tracks_active_opponent_instead_of_stationary_unassigned_slots` (opponent visibility). Existing tests were unchanged. This is not a claim that the complete repository suite passes.

## Physical model limits

Robot chassis remain planar; wheel climbing, pitch, roll, and lift are not represented. Robot and field side contacts use extruded planar faces rather than complete 3D top/edge contact manifolds. Bump support uses an analytic plane sampled at the sphere center, without rounded-edge continuous collision detection.

Adaptive substeps bound initial motion; they do not formally guarantee separation for acceleration generated by contacts or grossly overlapping initial states. Contact coefficients and launch parameters are illustrative and require calibration against field measurements. Dense pile timestep differences demonstrate why passing numerical tests alone does not establish real-world fidelity.


## Final measurements on legal startup fixtures

The final fixture generator filters fuel against expanded robot footprints and selects separated ball positions. Every standalone final scene reported **zero initial ball, robot, and floor penetration**. These standalone comparisons deliberately use a **flat field without bump regions or static field boxes**. Earlier sparse/settled stress fixtures could overlap parked bumpers; their algorithm comparisons are not evidence of legal field startup or normal peak fuel speeds. The pile fixture was a legal tangent stack throughout.

GPU 0, 24 worlds, 504 balls, six robots, 32 ticks × 3 repetitions, ten fixed 2 ms substeps:

| Path | Sparse | Pile |
| --- | ---: | ---: |
| Naïve fuel and robot all pairs | 9.077 ms | 8.520 ms |
| Selected cached fuel grid, robot direct scans | 7.488 ms | 7.249 ms |
| Fixed-cadence graph prototype | 5.631 ms | 6.212 ms |

Cached neighbors improved coupled physics by **1.21× in sparse fuel and 1.18× in the pile** versus matched all-pairs. The graph prototype reduced cached-path time a further **24.8% and 14.3%**, respectively, and short eager replay parity was exact in all six final repetitions. Graph results exclude production adaptive cadence and environment pointer rebinding, so graph replay remains disabled in the production runtime.

GPU 1 selected production configuration, legal flat fixtures, 64 ticks × 3 repetitions:

| Scene | Median per control tick | Actual substeps per tick | Ball penetration at end | Fuel energy at end |
| --- | ---: | ---: | ---: | ---: |
| Sparse | 8.149 ms | 10 throughout | 6.9 mm | 5.49 J/world |
| Pile | 11.701 ms | 10–32, mean approximately 14.59 | 14.2–17.6 mm | 294.6–296.1 J/world |

The pile started without penetration. Its adaptive increase is caused by subsequent contact-generated fuel motion rather than terrain or bumper spawn overlap. End-of-run maximum fuel speed was approximately 10.6–11.0 m/s; intermediate speed bounds required up to 32 substeps. This is a stress scenario, and coefficients still require physical calibration. The pile robot traveled approximately 1.472 m in 1.28 simulated seconds, versus approximately 2.30 m with no fuel, demonstrating sustained resistance through the coupled drivetrain.

The **full environment** comparison used production field geometry, seeded staged fuel, default controllers, action 7, observation materialization enabled, and info materialization disabled. It alternated static/realistic and realistic/static order across three repetitions, each measuring 32 steps after three warmup steps:

| Environment | Median step time |
| --- | ---: |
| Legacy fuel physics disabled | 61.476 ms |
| Selected moving fuel physics enabled | 69.390 ms |

This measured **1.129× runtime, or 12.9% additional environment-step time**. It is a short environment-loop comparison, not a full match, PPO training benchmark, or prediction for every contact density. Initialization and extension compilation are outside timing.

Final reports: `gpu0_final_graph_e2e.json` and `gpu1_default_adaptive.json`. Both report **no errors and unchanged source hashes during execution**. The full environment report fingerprints package Python and primary C++/CUDA/header sources, excluding generated HIP conversion files. Raw outputs remain on the HDD-backed project filesystem. Reports ending `_pre_spawnfix` or `_provisional` are retained diagnostics and are not final production evidence.
