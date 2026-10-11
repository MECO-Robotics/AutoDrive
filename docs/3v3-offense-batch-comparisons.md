# 3v3 offense batch comparisons

Use paired randomized batches to compare offense behavior changes from Codex
subagents. Keep candidate worktrees isolated and give each subagent a unique
branch and output label.

## Create candidate worktrees

Start each worktree from the same `dev` revision:

```sh
git worktree add ../AutoDrive-offense-a -b codex/offense-a dev
git worktree add ../AutoDrive-offense-b -b codex/offense-b dev
```

Have each subagent make one focused behavior change in its worktree. Record the
starting commit and avoid combining unrelated changes while comparing.

## Run the same randomized batch

Run the evaluator once in the baseline checkout and once in each candidate
worktree. Keep the seed, world count, ticks, and simulator settings identical.
For example:

```sh
python scripts/evaluate_3v3_offense_batch.py --label baseline --seed 64821 --envs 32 \
  --output outputs/3v3-offense-batches/baseline-64821.json
python scripts/evaluate_3v3_offense_batch.py --label offense-a --seed 64821 --envs 32 \
  --output outputs/3v3-offense-batches/offense-a-64821.json
python scripts/compare_3v3_offense_batches.py \
  outputs/3v3-offense-batches/baseline-64821.json \
  outputs/3v3-offense-batches/offense-a-64821.json \
  --output outputs/3v3-offense-batches/compare-a-64821.json
```

Increase the seed and repeat for additional batches before selecting a change.
The evaluator records per-world scores, acquisitions, source revision, timing,
and compact aggregate metrics. It does not produce per-match replay files.

The vector simulator uses one batch seed to initialize distinct world slots;
it does not accept independent seeds per slot. A scenario ID is therefore the
pair `(batch seed, slot)`. Matching batch settings across candidate worktrees
preserves the paired scenario layout. Inspect a fresh full replay for promising
candidates before merging their behavior change.

Generated artifacts are ignored by Git under `outputs/3v3-offense-batches/`.

## Results to date (2026-10-11)

The current comparison baseline is `3e996c9` (`Penalize pickup targets behind
obstacles`). Candidate runs use the known batch seeds `827104`, `827105`, and
`191873`, four world slots per seed, 8,000 ticks, fuel physics, replan interval
20, perception interval 2, and no teammate intent knowledge or coordination.
Baseline and candidate are run sequentially on the same GPU. Candidate decisions
must use only the acting robot's own observations/state. A positive aggregate
fuel delta is a useful acquisition signal, not by itself proof of generalization:
keep per-seed and per-world results and leave experimental branches unmerged
until their robustness is established.

The vector batch seed initializes four distinct slots; scenario IDs are
`seed:slot`. Identical seed/settings runs have shown appreciable world-level
variation, so small aggregate changes need repeated matched batches before they
are treated as robust. Each run must also be a representative end-to-end
simulation; this batch evaluator writes records but does not create dashboard
replays. If a replay is later used, first restart and verify the dashboard stack
against the current checkout, then generate a fresh replay.

### Behavior changes with positive paired evidence

| Change | Paired evidence | Disposition |
| --- | --- | --- |
| Turn intake toward fuel during a feasible approach (`407b00b`) | Against the prior raster-search behavior, the fixed 12-world comparison gained 462 acquisitions and 173 points. | In current `main`. |
| Rank fuel targets with robot momentum (`28e0802`) | Against the turn-envelope behavior, fixed 12-world comparison gained 295 acquisitions and 18 points. | In current `main`; modest score gain. |
| Penalize pickup targets behind obstacles (`3e996c9`) | Against momentum ranking, fixed 12-world comparison gained 887 acquisitions and 1,211 points; all three seed aggregates were score-positive. Historical mean score was 804.08, though only 5/12 worlds individually exceeded 800. | In current `main`; the clearest measured step toward the >800-point target. |
| Stagger no-track raster phases by robot slot | Historical comparison against an older raster baseline: +508 acquisitions, +15 points. Initial recheck: +58 acquisitions, +413 points across 12 worlds; seed fuel/score deltas 827104 +217/+370, 827105 +150/+161, 191873 −309/−118; fuel improved in 5/12 worlds. Two independent replications against current behavior baseline: A +741 fuel/+483 points (8/12 worlds fuel-positive, 8/12 score-positive; seed deltas +748/+551, −586/−648, +579/+580); B +796 fuel/+923 points (7/12 fuel-positive, 7/12 score-positive; seed deltas +232/+398, −132/−46, +696/+571). Replication mean scores: A 708.67→748.92, B 700.08→777.00. | Committed as `6d9b709` on `codex/no-track-raster-phase-recheck`; not merged. Both replications are aggregate-positive, but seed/world variance remains high and neither mean reached 800. Keep experimental pending broader robustness evidence. |

### Tested behavior changes that did not pass the fuel gate

| Change | Paired evidence | Disposition |
| --- | --- | --- |
| Raise stale fuel track age penalty from 0.4 to 0.8 | Against `3e996c9`: −1,417 acquisitions, −1,190 points; fuel improved in 4/12 worlds. | Attempt preserved as `0c105e1` on `codex/stale-track-age-08`; not merged. |
| Rank targets by moving-fuel interception before the top four | Against obstacle-aware ranking: −7,577 acquisitions, −4,445 points over 12 worlds. | Rejected by the acquisition/score evidence; branch `codex/moving-fuel-rank` remains experimental. |
| Stage behavior around FMS transitions (broad pre-staging) | −6,235 acquisitions, −1,410 points over 12 worlds. | Rejected; no merge. |
| Just-in-time FMS staging | −776 acquisitions, −963 points over 12 worlds. | Rejected; no merge. |
| Increase occluded-fuel velocity damping | −1,318 acquisitions, −621 points over 12 worlds. | Rejected; inference behind obstacles alone did not justify this damping change. |
| Stage loaded robots for the next HUB activation (`e33a265`, `codex/fms-activation-stage-current-main`; fixes `3e08901`, `7cb2c97`) | Fresh current-main paired batch: −7,950 acquisitions and −2,658 points; mean score 686.42→464.92. Seed fuel/score deltas: 827104 −2,486/−416, 827105 −1,872/−561, 191873 −3,592/−1,681. Fuel and score each improved in only 1/12 worlds. | Rejected. The candidate’s staging rule consistently degraded collection and scoring; remains committed on its branch, not merged. Earlier reported +763/+244 was not supported by the saved records and did not reproduce. |
| Abort current-HUB pickup when remaining score window is too short (`34b085a`, `codex/fms-active-window-abort-current-main`; parity fix `f1edd80`) | Fresh current-main paired batch: −1,161 acquisitions and −677 points; mean score 744.75→688.33. Seed fuel/score deltas: 827104 −927/−849, 827105 −113/+95, 191873 −121/+77. Fuel and score each improved in 5/12 worlds. Candidate runtime was 637.91s versus 677.52s baseline. | Rejected: faster execution did not preserve behavior quality; both fuel and aggregate score fell. Remains committed on its branch, not merged. |
| Weight each robot's cloud-edge fuel-track weights by its own observation confidence (`120de5a`, `codex/fuel-cloud-confidence`) | Against sequential `f4eae79`: +15 acquisitions but −243 points over 12 worlds; fuel improved in 6/12 worlds and score in 4/12. Seed fuel/score deltas: 827104 −252/−406, 827105 +56/+368, 191873 +211/−205. Mean score fell 704.42→684.17 and elapsed time rose 13.2%. | Mixed fuel-only result with lower scoring and high world variance; retained on its branch, not merged. Records are under `AutoDrive-fuel-cloud-confidence/evaluations/fuel-cloud-confidence/`. |
| Penalize pickup tracks near field boundaries (`7124128`, `codex/wall-clearance-pickup`) | Against sequential `f4eae79`: −321 acquisitions and −332 points over 12 worlds; mean score fell 745.08→717.42. Seed fuel/score deltas: 827104 −13/−151, 827105 −138/−180, 191873 −170/−1. | Rejected by both aggregate fuel and score evidence; attempt retained on branch, not merged. Per-world records are under `AutoDrive-wall-clearance/evaluations/fresh-local-candidate/`. |
| Defer loaded scoring while nearby alliance-zone fuel is reachable (`e83fc4c`, `codex/fuel-local-alliance-score`) | Against sequential `cc22785`: −982 acquisitions, +248 points over 12 worlds; only 2/12 worlds gained fuel. Mean score rose 645.25→665.92, still well below 800. Seed fuel/point deltas: 827104 −562/−227, 827105 −331/+126, 191873 −89/+349. Runtime increased 3.9%. | Fuel gate failed despite a small score gain; attempt retained on its branch, not merged. Records are under `AutoDrive-fuel-local-alliance-score/evaluations/fuel-local-alliance-score/`. |
| Estimate final intake turn time with bounded angular acceleration (`d4e986e`, `codex/intake-turn-acceleration`) | Against sequential `cc22785`: −675 acquisitions, −284 points over 12 worlds; mean score fell 686.25→662.58. Seed fuel/point deltas: 827104 −577/−489, 827105 +56/+128, 191873 −154/+77. Tick-loop runtime improved 7.3%. | Throughput improvement did not preserve behavior quality; fuel and score gates failed. Attempt retained on its branch, not merged. Records are under `AutoDrive-intake-turn-acceleration/evaluations/intake-turn-acceleration/`. |
| Prefer nearby dense fuel patches | +109 acquisitions but −136 points over 12 worlds. | Did not meet both acquisition and scoring goals; no merge. |
| Cluster pickup-throughput heuristic | −26 acquisitions, +924 points over 12 worlds. | Score-only improvement with a net fuel loss; not accepted as an offense gain. |

Earlier single-seed siloed-behavior branches also explored loaded return,
collection detours, lane search, reservations, score-zone choices, hysteresis,
stale-track extrapolation, and intent-gated coordination. Treat their branch
names as experiment history, not validated gains; do not reintroduce teammate
intent dependence into onboard choices. The current fixed-batch candidates are
committed to their own branches even when they lose, so code and measured result
remain linked. Candidate commits are not automatically merged to `main`.

### FMS action-window attempts on current main (2026-10-11)

Both candidates were ported by cherry-picking their behavior commits onto the
current `main` baseline (`65a55bf`) and compared on the same three batch seeds,
4 worlds each, 8,000 ticks, fuel physics, replan interval 20, perception
interval 2, and no teammate coordination. Baseline and candidate ran
sequentially on one assigned GPU; the two candidate experiments used separate
GPUs. The generated HIP source files from the shared checkout were copied into
the candidate worktrees so these paired evaluations used the same native
runtime sources. Records are outside Git in `evaluations/fms-stage-current-audit/`
and `AutoDrive-fms-window-pickup-eval/evaluations/fms-stage-current-audit/`,
and `evaluations/fms-window-current-audit/` and
`AutoDrive-fuel-velocity-gate/evaluations/fms-window-current-audit/`.

- **Loaded staging (`e33a265`, final tested code `7cb2c97`)**: −7,950
  acquisitions and −2,658 points overall; mean score 686.42→464.92. Per-seed
  fuel/score deltas: 827104 −2,486/−416, 827105 −1,872/−561, 191873 −3,592/−1,681.
  Per-world score/fuel deltas: 827104 `(-250,-740), (-113,-625), (+71,-421),
  (-124,-700)`; 827105 `(-35,-378), (-390,-894), (-126,-671), (-10,+71)`;
  191873 `(-452,-978), (-405,-905), (-516,-970), (-308,-739)`. Runtime fell
  from 703.17s to 610.67s alongside the large behavior loss. The historical
  branch initially failed evaluation due score-grid shape indexing; fixes are
  committed on the same branch and the final 12-world batch completed.
- **Active-window pickup abort (`34b085a`, final tested code `f1edd80`)**:
  −1,161 acquisitions and −677 points overall; mean score 744.75→688.33.
  Per-seed fuel/score deltas: 827104 −927/−849, 827105 −113/+95, 191873
  −121/+77. Per-world score/fuel deltas: 827104 `(-197,-103), (-97,-253),
  (-463,-496), (-92,-75)`; 827105 `(+121,+7), (+240,+307), (-393,-695),
  (+127,+268)`; 191873 `(-213,-331), (+122,+154), (-39,-220), (+207,+276)`.
  The first candidate attempt failed before writing records because a parity
  branch passed a Python bool to `torch.where`; fix `f1edd80` is included in the
  evaluated code. Runtime fell from 677.52s to 637.91s, but behavioral gates
  failed.

Neither FMS rule generalizes as an offense improvement on this batch. A prior
message reported the loaded-staging branch at +763 acquisitions/+244 points;
that result was not reproducible from the available records, and this clean
current-main paired comparison was strongly negative. Both candidate branches
remain unmerged. These were end-to-end simulator batch evaluations, not
 dashboard replays; no dashboard replay or stack verification was performed.

### Independent no-track phase replications (2026-10-11)

The phase-offset candidate was re-evaluated in two independent paired runs on
separate GPUs, using unchanged commit `6d9b709` and baseline behavior from
`main` (`8409867`, whose runtime behavior matches `3e996c9`). Each run used the
same three batch seeds and settings above, with baseline and candidate evaluated
sequentially on the assigned GPU. Both candidate batches completed normally.

- Replication A: +741 acquisitions and +483 points across 12 worlds; aggregate
  mean score 708.67→748.92. Seed fuel/score deltas were 827104 +748/+551,
  827105 −586/−648, and 191873 +579/+580. Fuel and score each improved in 8/12
  worlds. Per-world fuel/score deltas: 827104 `(+152,+101), (+566,+299),
  (+11,+138), (+19,+13)`; 827105 `(-405,-358), (-391,-387), (-127,-207),
  (+337,+304)`; 191873 `(+283,+159), (-278,-96), (+331,+266), (+243,+251)`.
  Candidate elapsed time was 773.09s versus 789.36s baseline (−2.1%, including
  cold-cache cost in the first seed).
- Replication B: +796 acquisitions and +923 points across 12 worlds; aggregate
  mean score 700.08→777.00. Seed fuel/score deltas were 827104 +232/+398,
  827105 −132/−46, and 191873 +696/+571. Fuel improved in 7/12 worlds and score
  improved in 7/12. Candidate elapsed time was 658.98s versus 721.79s baseline.

The aggregate direction reproduced, but per-seed/world differences are large;
mean candidate score remained below 800 in both replications. This is promising
but insufficient evidence to merge. JSON records are retained outside Git in
`AutoDrive-no-track-raster-phase-recheck/evaluations/no-track-phase-replication-a/`
and `AutoDrive-no-track-raster-phase-replication-b/evaluations/no-track-raster-phase-replication-b/`.
No dashboard replay or dashboard stack verification was performed; these
end-to-end batch records are the simulator evaluations for this behavior.

### Infrastructure and repeatability findings

- Reusing the captured computation graphs was faster than disabling graphs in
  the measured fixed batch. Graph-off was slower for all three seed batches and
  differed in world-level scores and acquisitions; it was not a lossless
  optimization. Keep the current graph path.
- A same-seed repeat on the same GPU differed substantially by world despite
  matching reset-state hashes. A short diagnostic first diverged around tick
  106 with velocity differences on the order of 1e-9 to 1e-8 before scoring or
  pickup. Duplicate-index `index_add_` contact reductions are a plausible
  source, but this has not been proven. Small behavior gains therefore need
  repeated paired batches before being called robust.
- The evaluator uses explicit seed resets, but that alone has not produced
  bit-identical long trajectories. Report seed-level aggregates and spread,
  not only a pooled win or one favorable scenario.

### Current candidate work

The latest fixed-batch attempts—fuel-cloud confidence, wall clearance,
local alliance scoring, and acceleration-aware intake turning—are complete and
preserved on their own branches. None established a robust improvement in both
fuel and scoring. Future candidates should start from current `main`, keep the
fixed batch unchanged, and add their branch hash and paired result here whether
they pass or fail. Never merge an experimental candidate automatically.
