# Neural architecture study: findings

**Completed:** 119 training cases, 15 paired CNN map evaluations, six instrumented absence references, and four fresh dashboard replays. All use frozen V5 source; the checkpoint/metadata audit passed. Known test and trajectory-repeat limitations are reported below.

A full-field CNN is useful enough to keep testing, but this early-learning screen does not establish a consistent advantage over the current scalar MLP. The implemented CNN combines a semantic field map with the current scalar vector; it does not replace the scalar stack. Capacity and extra retained-object information change together, so any performance difference cannot be attributed to convolution alone.

Seven alternatives were assigned to architecture-owning subagents: the existing MLP, residual MLP, Deep Sets, an entity Transformer, full-field CNN, residual full-field CNN, and a Fourier field network. All act through simulated perception, including opponent observations used by planning and avoidance. Both requested directions are implemented: learned defense against deterministic attackers and learned collection/scoring against a deterministic defender.

The Transformer is registered as `entity_attention`: two pre-normalized Transformer encoder blocks, each with four-head self-attention, a feed-forward layer and residual connections. A scalar context token reads the masked set of perceived entities; entity coordinates carry geometry without arbitrary sequence positions. It completed the same 15 main cases and two rate cases as every other architecture. This is an entity Transformer, rather than a field-patch vision Transformer.

## Early learning

The main comparison uses three training seeds per model/task, four held-out worlds per seed, and the same four PPO updates. Entries below are mean initial → fixed final evaluation. These are descriptive short-run outcomes, not convergence or statistical superiority claims.

| Network | Simple pickup / 6 ↑ | Scattered pickup / 24 ↑ | Focused defense: red points ↓ |
|---|---:|---:|---:|
| Existing MLP | 2.00 → 5.50 | 2.50 → 6.83 | 14.50 → 14.08 |
| Residual MLP | 5.25 → 4.50 | 6.25 → 8.00 | 14.50 → 6.00 |
| Deep Sets | 5.00 → 6.00 | 6.75 → 8.17 | 20.50 → 20.00 |
| Entity Transformer | 4.92 → 6.00 | 8.00 → 7.58 | 8.00 → 20.00 |
| Full-field CNN | 2.33 → 5.50 | 4.42 → 7.33 | 20.50 → 16.58 |
| Residual field CNN | 5.83 → 6.00 | 6.50 → 7.92 | 10.00 → 2.00 |
| Fourier field network | 4.42 → 6.00 | 6.67 → 7.08 | 8.00 → 14.50 |

The standard CNN matches the baseline's final simple-pickup mean and slightly exceeds its scattered-pickup mean. Their simple-pickup final seed vectors are identical: `[6, 4.5, 6]`. Deep Sets gives a useful inexpensive object-based comparison. Several models already start near the six-ball ceiling, making their smaller learning gains difficult to interpret.

The residual CNN has the best final focused-defense mean in this screen, but its improvement comes entirely from one seed: red points change `[24, 0, 6] → [0, 0, 6]`. The residual scalar MLP is also competitive. The standard CNN's final red points are `[24, 19.75, 6]`, versus baseline `[18.25, 0, 24]`; three seeds do not establish a reliable ranking. Use the fixed final checkpoint: selecting each model's best held-out intermediate evaluation would favor it after seeing the test results.

## Offense against a deterministic defender

At the same fixed final checkpoint, increased collection rarely becomes scoring:

| Network | Acquisitions initial → final ↑ | Blue points initial → final ↑ |
|---|---:|---:|
| Existing MLP | 3.42 → 8.00 | 0.08 → 0.00 |
| Residual MLP | 9.67 → 12.00 | 1.08 → 0.00 |
| Deep Sets | 8.17 → 11.58 | 0.00 → 0.00 |
| Entity Transformer | 8.50 → 9.33 | 0.83 → 1.25 |
| Full-field CNN | 4.25 → 11.08 | 2.83 → 0.00 |
| Residual field CNN | 9.75 → 10.33 | 1.17 → 0.00 |
| Fourier field network | 9.00 → 10.92 | 0.00 → 0.00 |

Attention's final scoring is confined to one training seed: `[0, 3.75, 0]`. The standard CNN and Fourier policy both score in some intermediate evaluations and return to zero at the fixed final checkpoint. Training itself supplies few blue-scoring events: totals across the three runs are 0/0/0/35/8/3/5 for baseline/residual/sets/attention/CNN/ResNet/Fourier. This is a weak test of learning to return and score, even when pickup improves.

## Learning-rate sensitivity

The additional sweep changes only the PPO learning rate on focused-defense training seed 7. Final red points, lower is better:

| Network | Initialized red points | 1e-4 | Main 3e-4 | 1e-3 |
|---|---:|---:|---:|---:|
| Existing MLP | 0 | 6 | 18.25 | 0 |
| Residual MLP | 0 | 24 | 12 | 0 |
| Deep Sets | 18 | 18 | 18 | 18 |
| Entity Transformer | 18 | 18 | 18 | 18 |
| Full-field CNN | 24 | 24 | 24 | 0 |
| Residual field CNN | 24 | 24 | 0 | 0 |
| Fourier field network | 0 | 0 | 0 | 0 |

Rates matter for some seeds, but these results do not establish an optimal rate. Several zero-score outcomes already occur at initialization; preserving them is not learned improvement. The 14 extra cases are kept separate from the 105 main three-seed cases.

## Does the standard CNN use the perceived-object map?

Across all five tasks and three final checkpoints, zeroing only the dynamic FUEL/opponent raster channels changes **0 of 3,120 paired greedy decisions**, with unchanged recorded acquisition/score/event metrics. Geometry, own/team/HUB channels and the current scalars remain available. This provides no observed greedy-decision benefit from those dynamic map channels in these frozen policies. It does not test removing the entire raster, retraining without it, stochastic training decisions, logits/value sensitivity, or the residual/Fourier encoders. The intervention is out of distribution and scalar redundancy remains.

Fresh normal evaluations exactly reproduce the main evaluation's metrics in all 15 cases and its complete saved fields in 14. Full-3v3 seed 43 is the exception: its first sampled difference is a 2.38e-7 m FUEL coordinate at tick160, growing to a maximum sampled robot-center displacement of 2.79 m and FUEL-coordinate difference of 1.10 m. Scores, event timing, action totals and sampled requested actions still agree. The main record lacks a complete per-decision trace, so full main-vs-fresh decision-sequence equality is not established. Normal-vs-masked trajectories also diverge in that case despite identical complete greedy action sequences; those physical differences cannot be attributed to map masking. The cause is unproven. Both records are retained, without retry.

See the [15-case aggregate](../evaluations/architecture-study-20261020/field-sensitivity-v5/aggregate.json) and [trajectory parity detail](../evaluations/architecture-study-20261020/field-sensitivity-v5/defense_3v3/seed-43/normal-parity-detail.json). This reproducibility limitation weakens trajectory-level conclusions; matching outcome metrics are a narrower claim.

## Fresh simulator replays

The dashboard services were restarted on V5 at 19:20:23 UTC; both were active, all eight checked HTTP endpoints returned 200, and all six served JavaScript files matched the checkout byte for byte. Four new replay records match their checkpoint and runtime hashes, are listed as current, and load through both playback API views. The older CNN replay is marked outdated.

| Replay / seed | Duration | Acquisitions by learned blue robots | Red / blue points |
|---|---:|---:|---:|
| CNN scattered pickup / 100019 | 4 s | 7 | 0 / 0 |
| CNN offense / 100019 | 12 s | 12 | 0 / 0 |
| CNN 3v3 transfer / 200019 | 160 s | 111 | 1,010 / 0 |
| Existing MLP 3v3 transfer / 200019 | 160 s | 65 | 835 / 0 |

Both full-match policies were trained only on the eight-second opening. They mostly choose collection or hold, never interception, and fail as full-match defenders here. All three learned robots eventually retain opponent tracks; sampled minimum opposing-center distances are about 0.95 m / 0.94 m. Both records include all five HUB-state transitions after the opening. This is one unseen-seed transfer check, not a long-horizon architecture ranking. Single-world replay sensor batching also differs from the four-world learning evaluations.

Open the [current dashboard](http://127.0.0.1:8765/) and select the saved scenario replay. Records:

- CNN pickup: `scenario-arch-full_field_cnn-d5d9a2afb689bd9a62db`
- CNN offense: `scenario-arch-full_field_cnn-ca7a1f6b1b5a7c27b845`
- CNN full match: `scenario-arch-full_field_cnn-b53282f0cbc9a0324778`
- Baseline full match: `scenario-arch-baseline_mlp-4817bff1c1d7e2a54546`

The [replay review record](../evaluations/architecture-study-20261020/replays-v5/review-summary.json) retains actual action counts, track validity, paths, HUB changes, outcomes and API checks.

![Single-seed transfer: points, retained opponent tracks, and sampled proximity](../evaluations/architecture-study-20261020/replays-v5/full-match-transfer.png)

## Scope and interpretation

The 4/8/12-second pickup/defense/offense horizons measure early learning before the first HUB phase change. The eight-second full-3v3 task is an opening probe: several policies receive no retained opponent observation in the sampled representative world. Similar opening scores do not establish similar defensive ability. Separate 160-second replays inspect later encounters and HUB phases, but are one-seed transfer examples for the baseline and standard CNN only.

The CNN sees a 12-channel, 32×64 full-field semantic raster with known geometry, own/team telemetry, and retained sensed FUEL/opponent tracks. It cannot see hidden opponents or unsensed FUEL. A raster cell with no retained object does not distinguish unobserved space from confirmed empty space. Policies retain eight tactical actions and only four collection candidates; detecting a distant useful cluster does not let the network select an arbitrary waypoint.

The offense reward combines collection and points. Acquisition gains must be reported separately from blue scoring; collecting more is not evidence of a successful collection-and-score cycle. The generic neural offense execution also lacks deterministic-only intake/scoring refinements. A scripted policy using the same neural execution path has scored in actual simulator qualification, so scoring is feasible, while reward exposure, credit assignment, and execution remain important limits on architecture conclusions.

Isolated tasks park inactive robot bodies outside the field, but the native wall solver can temporarily return them to a corner during contact substeps. This shared scenario limitation is documented. Additional instrumented absent-blue references check body exclusion at every contact boundary; their scores are descriptive context because removing observers changes perception batching and eager physics differs from graph execution.

Attention and the Fourier alternative remain eligible. These short runs do not rule out relational, global-spatial, recurrent, graph, state-space, or KAN approaches. A stronger next comparison would retrain matched-capacity scalar and field policies, independently vary map information, extend encounter-rich scoring horizons, and separate tuning seeds from final evaluation seeds.

## Validation and limitations

The final [119-case audit](../evaluations/architecture-study-20261020/v5-final-119-audit.json) passes source/checkpoint hashes, finite weights/losses, actual parameter updates, seeds/budgets, 20 ms physics, simulated perception and action masks. All 357 saved checkpoints were inspected. Architecture-owning subagents implemented and reviewed the alternatives; the current consolidated [owner review](../evaluations/architecture-study-20261020/owner-dense-review.md) retains raw seed values.

Focused tests: **55 passed**. Existing `test_tensor_3v3.py`: **7 passed, 1 failed**. The failing visibility assertion, `test_defender_tracks_active_opponent_instead_of_stationary_unassigned_slots`, also fails on isolated original HEAD `05f0364c9`; it remains a reported failure. This is not a clean whole-repository test run.

Actual simulator qualification includes offense and 3v3 hold at seed 100019, four worlds, 400 ticks, two executions each: recorded frames/events match bitwise. Scripted neural-path offense acquires `[5,6,11,8]` and scores blue `[0,3,5,6]`. Shared two-robot compatibility was exercised at seed731 for 200 ticks, with motion and final track counts `[38,14]`, no pickups/points. A same-process two-network offense CLI checks environment/graph lifetime separately.

The six proper absence references preserve setup and verify body exclusion at robot/FUEL contacts. Five repeats match bitwise; full-3v3 seed 100007 fails the preset 1e-5 trajectory tolerance with a 2.74e-5 m maximum recorded coordinate difference. Its scores/events match exactly and both runs remain saved without retry. This is separate from the larger seed 43 fresh-normal trajectory mismatch described above. No universal trajectory reproducibility claim is supported.

Current dashboard process/frontend/API verification and four reviewed fresh native records accompany the tests. See the [validation summary](../evaluations/architecture-study-20261020/validation-summary.json), [test records](../evaluations/architecture-study-20261020/tests-summary.json), and [absence review](../evaluations/architecture-study-20261020/references-absent-blue-v5-gpu/review-summary.json) for scenarios, seeds and exact outcomes.

## Evidence and reproduction

See the [primary-paper research review](NN_ARCHITECTURE_RESEARCH.md), [runbook](NN_ARCHITECTURE_RUNBOOK.md), and [complete recorded curves and per-seed tables](NN_ARCHITECTURE_RESULTS.md). The current runtime source hash is `bfa926060795a3c97422ab15f1cee94a15cd31ca9ce84c79cee1e1cd0a5be32f`.

Earlier comparisons were withdrawn for stale captured GPU pointers or indirect hidden-opponent inputs. They are retained as outdated records and contribute no numbers to this table. Current validation additionally checks complete commands under hidden-opponent changes, not only the network's direct tensors.
