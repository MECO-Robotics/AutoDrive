# Expected-yield cycle-rank fresh replication V

Status: preregistered before these six seeds were evaluated; behavior and seeds are frozen.

## Question and frozen behavior

Does the robot-local expected-fuel-yield-per-pickup-and-score-cycle rank from `a580691` generalize to fresh seed clusters, improve offense acquisitions without reducing score, and where does time go within the match?

- Baseline behavior: main commit `d43469414e972ca22b620e953d546a9a8d876211`.
- Candidate behavior: `a580691afb3237087943762a2aee86b3da012e36`, based directly on that baseline.
- Baseline evaluator checkout: `codex/expected-yield-cycle-rank-replication-baseline`, commit `5aa2cbf`.
- Candidate evaluator checkout: `codex/expected-yield-cycle-rank-replication-v`, commit `4ab387a`.
- The same batch evaluator and read-only trace recorder are committed in both evaluator checkouts. Their SHA-256 is `18672b6fe1608629e54d8462e512ed95c5941a496d7fb7cb2bb4abdc1358857a`.
- Candidate policy is unchanged from `a580691`: score eligible targets in the existing top-four list by confidence/age-weighted expected local fuel, clipped to the robot's free capacity, divided by estimated pickup-plus-return time. It uses the acting robot's measured velocity and local observations. No teammate intent, lane, reservation, or coordination is enabled.

## Fresh seeds and GPU assignment

The six seeds were generated before evaluation with `random.Random(2026101102).sample(range(100000, 1000000), 6)`, in this order:

`339752, 199448, 247246, 973688, 144124, 762406`.

Before registration, the full project tree was searched for these exact values in Markdown, JSON, CSV, text, log, Python, shell, and YAML records; no occurrences were found.

Two agents will run disjoint seed groups at the same time, one evaluator per AMD GPU:

| GPU | Fixed seed order |
| --- | --- |
| `cuda:0` | `339752, 199448, 247246` |
| `cuda:1` | `973688, 144124, 762406` |

For each seed, run baseline then candidate on the assigned GPU, back-to-back, using identical scenario and simulator settings. Do not alter the seed order, tune the candidate, remove seeds, or rerun a result because it is unfavorable. Preserve a failed process log and report any missing arm as incomplete.

## Simulator and trace settings

Each arm uses a four-world vector batch for 8,000 ticks at 0.02 seconds per tick, 504 fuel pieces, randomized scenarios, fuel physics enabled, fused sensor RNG, deterministic controllers, three offense and three defense robots, replan interval 20, perception interval 2, and no teammate coordination. Capture slot 0 in the same rollout every 10 ticks (800 samples through tick 8,000); the instrumented evaluator checks trace final score and acquisitions against that exact batch slot. Set `PYTORCH_ROCM_ARCH=gfx900`. Baseline and candidate extension caches are separate per GPU.

Run outputs and logs are written outside the source checkouts under `/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive-cycle-rank-replication-artifacts/` so evaluator revision and dirty-state provenance remain meaningful. Copy and commit all raw results, traces, logs, analysis, and this registration on the isolated records branch after the evaluations. Do not merge the behavior.

## Primary analysis

Pair exact world slots using scenario IDs. Report offense acquisitions as the sum for robot slots 0–2 and score as team 0 points. Report these separately:

- Per-world paired deltas and direction counts.
- Per-seed four-world means and deltas, treating the six seed batches as the independent clusters.
- Pooled acquisition totals and score mean, median, and number of worlds above 800 for each arm.
- A two-sided 95% Student-t interval across the six seed-cluster mean deltas (df=5), separately for fuel acquisitions per world and points per world.

Do not treat the 24 worlds as 24 independent seed clusters. A positive pooled fuel count is an attempt-level fuel gain only; promotion requires consistent seed evidence and score improvement. An interval crossing zero means the effect remains uncertain.

## Intra-match analysis

Analyze the traced slot 0 in each seed and arm, separately from the batch-wide primary totals. Compare robot-local traversal distance and speed, time below 0.3 m/s while over 1 m from the commanded motor target, progress while that target stays stable, exact perceived fuel-track assignment/switch/loss counts and dwell, intake-enabled and cluster-sweep time, acquisitions per intake-enabled minute, loaded time, loaded time during inactive HUB periods and active scoring-zone periods, and team score deltas during active and inactive HUB periods.

Fuel target IDs are perception track slots, not persistent physical piece IDs; report ID changes as target churn and do not claim that every change is a physical retarget. These sampled traces describe timing and state transitions; they do not establish causation from six worlds. The trace does not record collision impulses or hidden fuel positions. No dashboard replay is generated; the dashboard stack is therefore not part of this run.

## Decision and reporting rules

Keep this an isolated attempt regardless of outcome. Commit negative and inconclusive results too. If pooled candidate offense acquisitions exceed baseline, report the exact aggregate and seed-level deltas and update every available subagent as requested. Do not claim the >800 objective is met unless fresh-seed score evidence supports dependable performance; a single high score or positive acquisition total is insufficient.
