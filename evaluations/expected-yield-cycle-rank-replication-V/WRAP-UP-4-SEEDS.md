# Expected-yield cycle-rank replication V: quick closeout

Date: 2026-10-11 UTC  
Candidate: frozen robot-local expected-fuel-yield-per-pickup-and-score-cycle policy, commit a580691  
Baseline: d434694  
Disposition: positive pooled fuel gate, inconclusive generalization; keep candidate isolated.

## Run design and coverage

The preregistered campaign uses six fixed seed clusters: 339752, 199448, 247246, 973688, 144124, and 762406. The quick closeout finished four paired clusters (16 worlds per arm): 339752, 199448, 973688, and 144124. Seeds 247246 and 762406 were not started. No policy retuning occurred between seeds.

Each arm ran four worlds per seed for 8,000 ticks with fuel physics, replan interval 20, perception interval 2, and teammate coordination disabled. Every baseline/candidate pair used the same GPU and matching scenario IDs. The batch JSONs, per-world results, traces, and launch logs are retained alongside this report. Both evaluator processes have exited, and both GPUs are idle.

## Paired outcomes

| Seed | Baseline points | Candidate points | Delta points | Baseline offense acquisitions | Candidate offense acquisitions | Delta acquisitions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 339752 | 3,273 | 3,326 | +53 | 5,626 | 5,436 | −190 |
| 199448 | 2,782 | 2,440 | −342 | 4,825 | 4,728 | −97 |
| 973688 | 3,064 | 3,769 | +705 | 5,045 | 6,376 | +1,331 |
| 144124 | 2,791 | 2,961 | +170 | 4,817 | 5,325 | +508 |
| **Total (16 worlds)** | **11,910** | **12,496** | **+586** | **20,313** | **21,865** | **+1,552** |

Candidate median score was 809 versus 734.5 baseline; mean score was 781 versus 744.4. Nine of 16 candidate worlds and seven of 16 baseline worlds scored above 800. Fuel improved in two of four seed clusters and regressed in two; points improved in three clusters and regressed in one.

Across the four seed-cluster means, the candidate delta was +97.0 acquisitions and +36.6 points per world. Descriptive two-sided 95% Student-t intervals with df=3 are −181.7 to +375.7 acquisitions and −135.2 to +208.5 points per world. Both include zero. These are interim four-cluster intervals; the preregistered six-cluster analysis was not run. The evidence meets the positive pooled-fuel gate, but does not establish a general fuel or scoring gain.

Per-world score/acquisition deltas (candidate minus baseline), in slot order:

| Seed | Score deltas | Offense acquisition deltas |
| ---: | --- | --- |
| 339752 | +81, +26, +25, −79 | +72, −62, +83, −283 |
| 199448 | −280, +59, −54, −67 | −96, +91, −47, −45 |
| 973688 | +275, +233, +86, +111 | +537, +263, +278, +253 |
| 144124 | +251, −173, −77, +169 | +425, −142, +34, +191 |

## In-match observations

For each seed, slot 0 has an 800-frame trace sampled every ten ticks through tick 8,000. We inspected the traces and checked the final score and offense acquisition count against the corresponding batch record; all four pairs match exactly. These traces describe one world per seed, not all 16 worlds.

| Seed | Distance (m) | Low-speed/far time (s) | Targetward progress (m) | Command-target changes | Fuel-track switches | Intake time (s) | Acquisitions per intake min | Loaded active-score-zone time (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 339752 | +39.9 | −32.4 | +14.0 | −49 | +185 | +23.8 | −8.3 | −103.4 |
| 199448 | −131.6 | +122.6 | −29.8 | −318 | −42 | −39.0 | +19.3 | −99.0 |
| 973688 | +62.2 | −60.0 | +25.8 | +93 | +128 | +25.4 | +98.0 | +18.6 |
| 144124 | +23.5 | −39.2 | +8.9 | −10 | −84 | −34.4 | +128.5 | −13.8 |

Deltas are candidate minus baseline and aggregate the three offense robots in the traced world. Low-speed/far time counts time below 0.3 m/s while more than 1 m from the motor target. Three traces show more targetward progress and less low-speed/far time; seed 199448 moves in the opposite direction. Target switching and loaded time in the active scoring zone also vary by seed. This is useful diagnostic evidence, not a causal explanation. Fuel-track IDs are perception slots, not persistent piece identities.

## Validation and provenance

- End-to-end simulator batches: four baseline/candidate seed pairs, 4 worlds per arm per seed, 8,000 ticks per world. All eight batch invocations exited successfully.
- Trace inspection: 800 frames per trace, scenario ID matched the paired slot, and final score/acquisition counts matched batch JSON exactly.
- Teammate coordination: disabled in all runs.
- Unit/regression tests: none run. Dashboard replay: none generated, so dashboard currentness was not checked.
- The seeded evaluator JSONs capture expected frozen HEADs, but HIP JIT setup left identical local generated diffs in baseline and candidate source worktrees. The run records mark these revisions dirty. The matching generated diffs are one pickup-grid declaration argument removal in tensor_3v3_pickup_grid_hip_hip.cpp and trailing-space changes in tensor_collision_multi_hip_kernel.hip and tensor_fuel_candidate_rank_kernel.hip. No manual source edit or cleanup was performed in those worktrees.
- The six-seed preregistered confidence analysis is incomplete because 247246 and 762406 were not run.

## Infrastructure and promotion decision

The world-parallel robot-contact prototype remains at commit db0630b on codex/multiworld-contact-parallel. A static path audit was completed, but compiled serial-versus-parallel bitwise parity and end-to-end same-seed performance checks were not run. No lossless infrastructure gain has been established.

At the time of this experimental closeout, the policy remained isolated because its generalization evidence was inconclusive. The user later directed merging the policy and its trace instrumentation into the default main worktree to update the existing live dashboards. No infrastructure branch was merged. The merge is not evidence of a statistical promotion. Prior attempt context, including why a single score above 800 was insufficient, is summarized in docs/3v3-offense-batch-comparisons.md and in the quick closeout recorded at commit 2e33454 on codex/offense-goal-wrap-up.
