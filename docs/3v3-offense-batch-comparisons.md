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
| Stagger no-track raster phases by robot slot | Historical comparison against an older raster baseline: +508 acquisitions, +15 points. Recheck against current `main`: +58 acquisitions, +413 points across 12 worlds; seed deltas 827104 +217/+370, 827105 +150/+161, 191873 −309/−118; fuel improved in 5/12 worlds. | Committed as `6d9b709` on `codex/no-track-raster-phase-recheck`; not merged. Narrow, mixed result pending repeatability check. |

### Tested behavior changes that did not pass the fuel gate

| Change | Paired evidence | Disposition |
| --- | --- | --- |
| Raise stale fuel track age penalty from 0.4 to 0.8 | Against `3e996c9`: −1,417 acquisitions, −1,190 points; fuel improved in 4/12 worlds. | Attempt preserved as `0c105e1` on `codex/stale-track-age-08`; not merged. |
| Rank targets by moving-fuel interception before the top four | Against obstacle-aware ranking: −7,577 acquisitions, −4,445 points over 12 worlds. | Rejected by the acquisition/score evidence; branch `codex/moving-fuel-rank` remains experimental. |
| Stage behavior around FMS transitions (broad pre-staging) | −6,235 acquisitions, −1,410 points over 12 worlds. | Rejected; no merge. |
| Just-in-time FMS staging | −776 acquisitions, −963 points over 12 worlds. | Rejected; no merge. |
| Increase occluded-fuel velocity damping | −1,318 acquisitions, −621 points over 12 worlds. | Rejected; inference behind obstacles alone did not justify this damping change. |
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
