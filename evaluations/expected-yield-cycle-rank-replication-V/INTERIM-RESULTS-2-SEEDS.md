# Expected-yield cycle-rank replication V: interim paired results

Updated 2026-10-11 UTC. The frozen six-seed campaign is ongoing. This report covers only the first two completed seed clusters; it is not the preregistered final analysis.

## Paired end-to-end outcomes

Each arm used four worlds, 8,000 ticks, fuel physics, deterministic controllers, replan interval 20, perception interval 2, and no teammate coordination. Seeds and world slots pair exactly. Acquisition counts below sum offense robots 0–2.

| Seed | Arm | Team-0 score by world | Offense acquisitions by world | Seed total score | Seed total acquisitions |
| ---: | --- | --- | --- | ---: | ---: |
| 339752 | Baseline | 731, 814, 990, 738 | 1263, 1485, 1448, 1430 | 3273 | 5626 |
| 339752 | Candidate | 812, 840, 1015, 659 | 1335, 1423, 1531, 1147 | 3326 | 5436 |
| 973688 | Baseline | 531, 640, 900, 993 | 898, 1115, 1415, 1617 | 3064 | 5045 |
| 973688 | Candidate | 806, 873, 986, 1104 | 1435, 1378, 1693, 1870 | 3769 | 6376 |

The pooled eight-world delta is **+1,141 offense acquisitions** (+142.6/world) and **+758 points** (+94.8/world). Candidate median score is 856.5 versus 776 baseline; 7/8 candidate worlds and 4/8 baseline worlds scored above 800. The cluster-level fuel direction is mixed: seed 339752 lost 190 acquisitions while seed 973688 gained 1,331. Score improved in both clusters. With only two independent seed clusters, these are promising but inconclusive results; no registered six-cluster confidence interval is reported here.

## Same-rollout trace observations

Both arms produced 800 trace frames for slot 0, sampled every ten ticks through tick 8,000. Trace scenario IDs matched the batch slot, and final score plus per-robot acquisition counts matched exactly.

| Seed | Candidate-minus-baseline distance | Low-speed/far time | Targetward progress | Command-target changes | Fuel-track switches | Intake-enabled time | Acquisitions / intake minute | Loaded active-score-zone time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 339752 | +39.9 m | −32.4 s | +14.0 m | −49 | +185 | +23.8 s | −8.3 | −103.4 s |
| 973688 | +62.2 m | −60.0 s | +25.8 m | +93 | +128 | +25.4 s | +98.0 | +18.6 s |

In these two traced worlds the candidate moved farther, made more progress toward commanded targets, and spent less time below 0.3 m/s while more than 1 m from its motor target. It also changed perceived fuel-track selection more often. Intake throughput and loaded scoring-zone time differed by seed, as did fuel gain. This is descriptive evidence from one traced world per seed; fuel-track IDs are perception slots, not persistent piece identities, and these traces do not establish a general causal mechanism.

## Provenance and next step

Raw batch JSON, traces, and resume logs are stored beside this report under `baseline/`, `candidate/`, `trace/`, and `logs/`. The original compilation-interruption logs remain preserved separately. Both arms report their expected frozen source commits and no coordination. HIP initialization left identical local source diffs in the baseline and candidate evaluator worktrees; the files differ only by one pickup-grid declaration argument removal and trailing whitespace in two launcher calls. The paired batches used the same runtime source state; their output revision records correctly mark the checkouts dirty.

Continue the remaining fixed seed order without tuning: baseline/candidate 199448 and 247246 on GPU 0; baseline/candidate 144124 and 762406 on GPU 1. Report each seed separately, then run the preregistered six-cluster paired analysis. Keep the candidate on its experiment branch; do not merge it to main based on this interim result.
