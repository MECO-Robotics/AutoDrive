# Expected fuel yield per pickup-score cycle

## Hypothesis and behavior

The deterministic offense picker currently ranks the four nearest locally eligible fuel tracks by travel distance with small age and obstacle penalties. A nearby isolated piece can therefore outrank a slightly farther cluster even when the robot can carry only some of that cluster and must return to score its load.

For each existing top-four candidate, this branch estimates an opportunity yield from the acting robot's eligible local fuel tracks within one chassis diameter of the candidate. Each eligible track index contributes at most once per candidate, weighted by perception confidence with an age decay; the total is clipped to that robot's remaining carrying capacity. It estimates pickup travel time using the robot's current velocity projected toward the candidate, acceleration, speed and turn-rate limits, then adds a rest-start direct return-time estimate from the candidate to the robot's own alliance scoring-zone boundary. Intake cooldown, one intake interval, and the existing track-age and obstacle penalties are included. During an active HUB, the picker minimizes estimated cycle time divided by expected yield. During an inactive HUB, existing staging-distance ranking remains unchanged.

The ranking reads only the acting robot's own pose, velocity, cooldown, capacity and eligible local track set, plus static game geometry and alliance scoring-zone geometry. It does not use teammate state, intent, lanes, reservations, or coordination. Pickup eligibility, action selection, and movement commands are unchanged.

## Limitations

This is a local opportunity-density estimate, not a route planner or a guarantee that nearby tracks can all be collected. It only reranks the existing top four candidates, so the best cluster can be excluded by the current nearest-track shortlist. The cycle model assumes a direct route, starts the return from rest, and does not predict future pile movement, opponent pressure, HUB transitions, or bump detours beyond the existing distance penalty. Confidence and age decay are approximate collection probabilities. The hypothesis needs paired unseen-seed simulator evaluation and should not be combined with other strategy branches without a separate composition holdout.

## Validation status

The frozen policy was evaluated against baseline d434694 in four paired seed clusters: 339752, 199448, 973688, and 144124. Each arm ran four worlds for 8,000 ticks with fuel physics, replan interval 20, perception interval 2, and teammate coordination disabled. The candidate gained 1,552 offense acquisitions and 586 points across 16 worlds; its median score was 809 versus 734.5 baseline. Fuel improved in two clusters and regressed in two. Four-cluster 95% intervals include zero, and preregistered seeds 247246 and 762406 were not run. This is a positive pooled result but inconclusive evidence of generalization. The batch JSONs and 800-frame slot-0 traces are archived under evaluations/expected-yield-cycle-rank-replication-V/; trace scores and acquisitions matched the batch records.

No unit or regression tests ran, and no dashboard replay was generated. The candidate was merged into the default main worktree at the user's direction after closeout; that deployment does not change the inconclusive evaluation result.
