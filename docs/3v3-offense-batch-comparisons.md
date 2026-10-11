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

### At a glance

- **Best established change:** obstacle-aware pickup ranking (`3e996c9`) is
  part of `main`. Its original 12-world paired batch gained 887 acquisitions
  and 1,211 points over momentum ranking, with score-positive seed aggregates
  on all three seeds. The historical candidate mean was 804.08, but only 5/12
  worlds exceeded 800.
- **Promising but mixed, unmerged:** no-track raster phase staggering has
  six current-main holdouts/replications: four gained aggregate fuel and
  points, while F and J regressed. Across these six batches it gained 3,621
  fuel and 3,713 points over 72 worlds. Its F mean score fell 778.67→768.00;
  an earlier E mean reached 822.25. Retain it as a selectable strategy, but
  do not claim every holdout generalizes positively.
- **FMS local action window, mixed and unmerged:** fresh holdout J gained 364
  fuel but only 10 points over 12 worlds; mean score was 757.83→758.67, with
  the result driven by two positive seeds and one large negative seed. Keep it
  isolated pending further unseen-seed evidence.
- **New speed-lookahead trial:** nominally gained 16 acquisitions and 184
  points, but fuel-positive evidence was mixed across seeds and close to the
  observed run variance. Keep it experimental and unmerged.
- **Target hysteresis, unmerged:** the initial batch and holdout F gained
  fuel/points, but fresh holdout G lost 405 fuel and 688 points. Across three
  12-world batches it is still +519 fuel/+1,509 points, with combined mean
  score 715.50→757.42. Keep the branch as an alternate with mixed evidence.
- **Not accepted:** the current-main FMS staging and stale-track extrapolation
  candidates failed their aggregate gates; other tested ranking, confidence,
  boundary, turn-cost, scorer-aim, and pickup-braking changes are itemized below
  with their paired results.
- **Attempt history:** behavior attempts stay committed on their own branches,
  including failed runs and evaluation fixes. Only measured behavior changes
  are candidates for promotion; none of the current experimental branches is
  merged into `main`.

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
| Stagger no-track raster phases by robot slot | Historical comparison against an older raster baseline: +508 acquisitions/+15 points. Initial recheck: +58/+413. Current-main replications A/B: +741/+483 and +796/+923. Holdout D: +871/+799. E: +2,190/+1,965, mean 658.50→822.25. F: −104/−128, mean 778.67→768.00. Across the five current-main batches A/B/D/E/F (60 worlds): +4,494 fuel/+4,042 points. | Committed as `6d9b709` on `codex/no-track-raster-phase-recheck`; not merged. Four batches gained fuel and points; F regressed. Keep as a separate selectable strategy; accumulated evidence is positive, though individual holds remain mixed. |
| Retain a robot's selected fuel target while it remains competitive, scaling the margin by its own targetward motion (`14f1ce4`, `codex/target-hysteresis-current`) | Initial current-main batch: +165 fuel/+1,030 points. Holdout F: +759/+1,167, mean 702.08→799.33. Holdout G: −405/−688, mean 753.08→695.75. Across all three 12-world batches: +519 fuel/+1,509 points; combined mean 715.50→757.42. | Mixed generalization: pooled aggregate fuel/score remain positive, but holdout G regressed strongly and none of these combined batches reached 800. Preserve the separate strategy branch; do not merge. |
| Scale no-track raster waypoint lookahead with the acting robot's speed (`470adef`, `codex/raster-speed-lookahead`) | Current-main paired batch: +16 acquisitions and +184 points across 12 worlds; acquisition-positive in 8/12 worlds and score-positive in 7/12. Seed acquisition/score deltas: 827104 −13/−94, 827105 −121/+199, 191873 +150/+79. Mean score 728.08→743.42; candidate elapsed time 594.62s versus 718.98s baseline. | Nominal aggregate gain is within observed run variance and seed results are mixed. Keep committed on its branch, experimental and unmerged. End-to-end records are in `AutoDrive-raster-speed-lookahead/evaluations/raster-speed-lookahead/` and `evaluations/raster-speed-lookahead-baseline/`. |

### Tested behavior changes that did not pass the fuel gate

| Change | Paired evidence | Disposition |
| --- | --- | --- |
| Raise stale fuel track age penalty from 0.4 to 0.8 | Against `3e996c9`: −1,417 acquisitions, −1,190 points; fuel improved in 4/12 worlds. | Attempt preserved as `0c105e1` on `codex/stale-track-age-08`; not merged. |
| Rank targets by moving-fuel interception before the top four | Against obstacle-aware ranking: −7,577 acquisitions, −4,445 points over 12 worlds. | Rejected by the acquisition/score evidence; branch `codex/moving-fuel-rank` remains experimental. |
| Stage behavior around FMS transitions (broad pre-staging) | −6,235 acquisitions, −1,410 points over 12 worlds. | Rejected; no merge. |
| Just-in-time FMS staging | −776 acquisitions, −963 points over 12 worlds. | Rejected; no merge. |
| Increase occluded-fuel velocity damping | −1,318 acquisitions, −621 points over 12 worlds. | Rejected; inference behind obstacles alone did not justify this damping change. |
| Extrapolate selected stale fuel position by observed velocity and age confidence (`06b1f12`, `codex/stale-fuel-extrapolation-current`) | Two independent paired batches: A +175 points/−152 acquisitions over 12 worlds; B −323/−303. Across 24 worlds: −148 points/−455 acquisitions. Seed score/fuel deltas A: 827104 +185/+62, 827105 −488/−761, 191873 +478/+547; B: −490/−558, −18/+305, +185/−50. | Mixed and not generalizable; fails the fuel gate across both replications. Keep its attempt branch, do not merge. Records are in `AutoDrive-stale-fuel-extrapolation-current/evaluations/stale-fuel-extrapolation-current/`. |
| Stage loaded robots for the next HUB activation (`e33a265`, `codex/fms-activation-stage-current-main`; fixes `3e08901`, `7cb2c97`) | Fresh current-main paired batch: −7,950 acquisitions and −2,658 points; mean score 686.42→464.92. Seed fuel/score deltas: 827104 −2,486/−416, 827105 −1,872/−561, 191873 −3,592/−1,681. Fuel and score each improved in only 1/12 worlds. | Rejected. The candidate’s staging rule consistently degraded collection and scoring; remains committed on its branch, not merged. Earlier reported +763/+244 was not supported by the saved records and did not reproduce. |
| Abort current-HUB pickup when remaining score window is too short (`34b085a`, `codex/fms-active-window-abort-current-main`; parity fix `f1edd80`) | Fresh current-main paired batch: −1,161 acquisitions and −677 points; mean score 744.75→688.33. Seed fuel/score deltas: 827104 −927/−849, 827105 −113/+95, 191873 −121/+77. Fuel and score each improved in 5/12 worlds. Candidate runtime was 637.91s versus 677.52s baseline. | Rejected: faster execution did not preserve behavior quality; both fuel and aggregate score fell. Remains committed on its branch, not merged. |
| Measure midfield fuel proximity from the acting robot's forward intake reference (`9a8a24e`, `codex/intake-proximity-return`) | Current-main paired batch: −600 acquisitions and −446 points across 12 worlds; mean score 744.83→707.67. Seed fuel/score deltas: 827104 −442/−336, 827105 −508/−91, 191873 +350/−19. Fuel improved in 4/12 worlds and score in 3/12. Candidate runtime was 728.60s versus 699.91s baseline. | Rejected by both aggregate fuel and score evidence; committed on its branch, not merged. Records are in `evaluations/intake-proximity-return/baseline/` and `AutoDrive-intake-proximity-return/evaluations/intake-proximity-return/candidate/`. |
| Use a continuous stopping controller for HUB scoring aim (`08f39ad`, `codex/score-aim-continuous`) | Current-main paired batch: −358 acquisitions and −576 points across 12 worlds; mean score 743.08→695.08. Seed fuel/point deltas: 827104 −529/−499, 827105 +114/−58, 191873 +57/−19. Fuel improved in 7/12 worlds, score in 5/12. Runtime was 655.74s baseline and 573.99s candidate. | Failed both aggregate gates; faster runtime did not preserve scoring. Keep committed on its branch, not merged. Records are in `evaluations/score-aim-continuous/baseline/` and `AutoDrive-score-aim-continuous/evaluations/score-aim-continuous/candidate/`. |
| Cap direct pickup fallback speed with a robot-local braking envelope (`6592633`, `codex/pickup-braking-envelope`) | Current-main paired batch: −1,302 acquisitions and −934 points; mean score 729.75→651.92. Seed fuel/score deltas: 827104 −1,324/−974, 827105 +174/+373, 191873 −152/−333. Fuel and score each improved in 4/12 worlds. Runtime was 654.67s baseline and 709.63s candidate. | Failed both aggregate gates and increased runtime. Keep committed on its branch, not merged. Records are in `evaluations/pickup-braking-envelope/baseline/` and `AutoDrive-pickup-braking-envelope/evaluations/pickup-braking-envelope/candidate/`. |
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

### Stale fuel position extrapolation replications (2026-10-11)

The prior single-seed change `70c5784` estimated a selected stale fuel track's
current position from its last observed position, velocity, and age with an
exponential confidence decay. It was ported onto the current-main behavior
baseline as `06b1f12` and run in two independent paired replications. Each used
seeds 827104, 827105, and 191873, four worlds per seed, 8,000 ticks, fuel
physics, replan interval 20, perception interval 2, and no teammate
coordination. Baseline and candidate ran sequentially on each assigned GPU.

- Replication A: +175 points, −152 acquisitions; mean score 651.17→665.75.
  Per-seed score/fuel deltas: 827104 +185/+62, 827105 −488/−761, 191873
  +478/+547. Per-world score/fuel deltas: 827104 `(+333,+287), (-139,-282),
  (+76,+223), (-85,-166)`; 827105 `(+61,+186), (-321,-330), (-188,-422),
  (-40,-195)`; 191873 `(-75,+25), (+420,+498), (-45,-54), (+178,+78)`.
  Runtime was 681.97s baseline and 716.51s candidate.
- Replication B: −323 points, −303 acquisitions; mean score 702.17→675.25.
  Per-seed score/fuel deltas: 827104 −490/−558, 827105 −18/+305, 191873
  +185/−50. Per-world score/fuel deltas: 827104 `(+122,+240), (-83,-59),
  (-479,-656), (-50,-83)`; 827105 `(+260,+276), (+53,+241), (+31,+97),
  (-362,-309)`; 191873 `(+114,-6), (+99,+125), (+22,+107), (-50,-276)`.
  Runtime was 677.01s baseline and 658.14s candidate.

Across both replications, the candidate lost 455 acquisitions and 148 points
in 24 worlds. Its direction changed across runs and seeds, so stale-position
extrapolation did not demonstrate robust behind-obstacle inference value. The
branch remains committed and unmerged. JSON batch records are retained in
`AutoDrive-stale-fuel-extrapolation-current/evaluations/stale-fuel-extrapolation-current/`.
No dashboard replay or stack verification was performed; these are simulator
batch records.

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

### Corrected independent phase holdout H (2026-10-11)

The first recorded candidate run for Holdout H was invalid: its evaluator
process imported `frc_defense` from the main checkout through the shared
editable install, even though it ran from the phase worktree. The earlier
candidate records remain on `codex/no-track-raster-phase-recheck` as historical
attempts and are explicitly superseded there. The corrected run set `PYTHONPATH`
to the phase candidate worktree and verified the imported package and behavior
module paths before evaluation. It tested the same seeds and settings as the
GPU0 main baselines above: 615379, 704167, and 992791; four worlds each; 8,000
ticks; fuel physics; replan interval 20; perception interval 2; no teammate
intent or coordination.

The corrected phase-only candidate gained 1,229 fuel acquisitions and 1,223
points over 12 paired worlds. Fuel improved in 10/12 worlds and points in
8/12. Mean team-0 score rose from 634.75 to 736.67, still below 800. Per-seed
fuel/point deltas were 615379 +730/+584, 704167 +79/+137, and 992791 +420/+502.
The code remains committed separately at `6d9b709`; corrected provenance,
records, and the invalid-attempt correction are committed on that branch at
`e53c2b3`. This result is positive aggregate evidence, not a merge: phase
staggering remains an optional branch strategy and is not bundled with
hysteresis or lane search. Records are in
`AutoDrive-no-track-raster-phase-recheck/evaluations/phase-holdout-H-candidate-corrected-gpu0/`.
No dashboard replay was generated or checked.

### Pre-registered independent Holdout I (2026-10-11)

Before running either candidate, reserve batch seeds 712943, 835117, and
264397 (four worlds per seed). Exact-seed searches found no prior occurrence
in the tracked evaluation JSON or comparison docs. Compare two strategies
separately against device-matched current-main baselines: phase staggering
alone (`6d9b709`) on `cuda:0`, and FMS action-window filtering alone
(`e016b22`) on `cuda:1`. Both use 8,000 ticks, fuel physics, replan interval
20, perception interval 2, and no teammate intent or coordination. Do not
combine the candidates in this holdout. Preserve the branch and complete
paired records even when results regress; report per-seed as well as pooled
fuel and score deltas. Dashboard replay is not part of this batch plan.

Both candidate arms completed with imports pinned to their separate
worktrees; all candidate and baseline scenario IDs/settings matched. The
frozen phase strategy (`6d9b709`, result commit `437a703`) gained 664 fuel
acquisitions and 483 points over 12 worlds; mean score rose from 761.33 to
801.58. Per-seed fuel/point deltas were 712943 +436/+107, 835117 −154/−46,
and 264397 +382/+422. Fuel and score each improved in 7/12 worlds, and
three-seed cluster intervals cross zero. The separate FMS strategy (`e016b22`,
result commit `d953ec9`) gained 1,431 acquisitions and 957 points; mean score
rose from 757.67 to 837.42. Its per-seed fuel/point deltas were 712943
+570/+505, 835117 +802/+416, and 264397 +59/+36; fuel improved in 8/12 worlds
and points in 9/12.

Each arm cleared 800 on Holdout I, but Holdout H and I pooled means remain
below 800 for both: phase 698.04→769.13 across 24 worlds, FMS 696.21→759.29.
Keep both strategies separate and unmerged; the I result is positive evidence
that merits another unseen-seed replication, not proof that either strategy
generalizes above target. Candidate records and per-world analyses are on the
respective branches in `AutoDrive-no-track-raster-phase-recheck/evaluations/`
and `AutoDrive-fms-local-action-window/evaluations/`.

### Pre-registered independent Holdout J (2026-10-11)

Reserve seeds 153719, 246817, and 390527 (four worlds each) for one more
independent replication; exact-seed searches found no existing occurrence in
the comparison docs or evaluation JSONs. Repeat only the two standalone
strategies, phase staggering on `cuda:0` and FMS action-window filtering on
`cuda:1`, each paired sequentially with current-main baseline on its assigned
device. Settings remain 8,000 ticks, fuel physics, replan interval 20,
perception interval 2, and no teammate intent or coordination. Candidate
imports must be pinned and verified against their worktrees. Keep both
branches separate, retain all outcomes, and report per-world, per-seed, and
pooled fuel/score deltas. This batch is a replication, not a tuning set.

### Phase-stagger + FMS action-window bundle Holdout H (2026-10-11)

Because the phase-stagger and FMS-window edits affect separate sections of
the action-selection flow, they were cherry-picked into a separate bundle
branch (`codex/phase-fms-bundle`, commits `2d4fa18` and `6390dcd`) and tested
together without changing either standalone branch. The bundle was paired
against main revision `60b2aaa` on seeds 615379, 704167, and 992791, four
worlds per seed, with the same 8,000-tick, fuel-physics, replan-20,
perception-2, no-teammate-coordination settings. Candidate imports were pinned
to the bundle worktree and verified before launch.

Across 12 worlds, the bundle gained 831 acquisitions and 984 points; mean
score rose from 634.75 to 716.75, below 800. Seed acquisition/point deltas
were +727/+585, +153/+371, and −49/+28. Acquisitions improved in 9/12 worlds
and points in 11/12. On this shared holdout, the bundle aggregate was below
phase-only (+1,229 acquisitions/+1,223 points) and FMS-only (+1,001/+557).
These are separate candidate comparisons against common baselines, not a
factorial estimate of interaction. Keep the bundle as its own unmerged
strategy branch. Per-world outcomes, import provenance, and evaluator records
are committed at `bb797e8` under
`AutoDrive-phase-fms-bundle/evaluations/phase-fms-bundle-holdout-H-gpu0/`.
No dashboard replay was generated or checked.

### Intake proximity and speed-lookahead trials (2026-10-11)

Both new candidates were committed before evaluation, started from current-main
runtime behavior, and tested on seeds 827104, 827105, and 191873 with four
worlds per seed, 8,000 ticks, fuel physics, replan interval 20, perception
interval 2, and no teammate coordination. Each baseline/candidate pair ran
sequentially on one GPU; the two candidate experiments ran concurrently on
separate GPUs. Their batches wrote end-to-end simulator records, not dashboard
replays, so no dashboard replay or stack verification was performed.

- **Forward-intake proximity (`9a8a24e`, `codex/intake-proximity-return`)**:
  −600 acquisitions and −446 points; mean score 744.83→707.67. Per-seed
  fuel/score deltas: 827104 −442/−336, 827105 −508/−91, and 191873 +350/−19.
  Per-world fuel/score deltas: 827104 `(-150,-47), (-365,-357), (+382,+302),
  (-309,-234)`; 827105 `(-144,-39), (+147,+105), (-90,-28), (-421,-129)`;
  191873 `(+266,-87), (-146,-263), (-264,-12), (+494,+343)`. Runtime was
  699.91s baseline and 728.60s candidate. The one-line local geometry change
  did not generalize across the batch; branch remains committed and unmerged.
- **Speed-scaled no-track lookahead (`470adef`,
  `codex/raster-speed-lookahead`)**: +16 acquisitions and +184 points; mean
  score 728.08→743.42. Per-seed fuel/score deltas: 827104 −13/−94, 827105
  −121/+199, and 191873 +150/+79. Per-world fuel/score deltas: 827104
  `(+79,-103), (-108,-84), (+18,+41), (-2,+52)`; 827105 `(-491,-307),
  (+94,+310), (+30,+73), (+246,+123)`; 191873 `(+1,-53), (+377,+410),
  (+209,+99), (-437,-377)`. Runtime was 718.98s baseline and 594.62s
  candidate. The small net fuel gain is within observed run variance; keep it
  experimental and unmerged.

JSON records are retained in `evaluations/intake-proximity-return/` and
`evaluations/raster-speed-lookahead-baseline/` in the baseline checkout and in
the respective candidate worktrees under
`AutoDrive-intake-proximity-return/evaluations/intake-proximity-return/` and
`AutoDrive-raster-speed-lookahead/evaluations/raster-speed-lookahead/`.

### Scoring aim and pickup braking trials (2026-10-11)

Both candidates were committed before evaluation and paired against runtime
behavior from `main` (`0141e24`). They used the fixed 3-seed x 4-world protocol,
8,000 ticks, fuel physics, replan interval 20, perception interval 2, and no
teammate coordination. Each baseline/candidate pair ran sequentially on one GPU;
the two experiments used separate GPUs. These were end-to-end simulator
records, not dashboard replays; no dashboard replay or stack verification was
performed.

- **Continuous HUB scoring aim (`08f39ad`, `codex/score-aim-continuous`)**:
  −358 acquisitions and −576 points; mean score 743.08→695.08. Per-seed
  fuel/score deltas: 827104 −529/−499, 827105 +114/−58, 191873 +57/−19.
  Per-world fuel/score deltas: 827104 `(-174,-183), (+16,+34), (-133,-84),
  (-238,-266)`; 827105 `(+106,+28), (+31,-60), (+53,+15), (-76,-41)`;
  191873 `(+202,+149), (-368,-380), (+4,-6), (+219,+218)`. Runtime was
  655.74s baseline and 573.99s candidate. Candidate code did not improve the
  aggregate acquisition or scoring result and remains committed, unmerged.
- **Robot-local pickup braking envelope (`6592633`,
  `codex/pickup-braking-envelope`)**: −1,302 acquisitions and −934 points;
  mean score 729.75→651.92. Per-seed fuel/score deltas: 827104 −1,324/−974,
  827105 +174/+373, 191873 −152/−333. Per-world fuel/score deltas: 827104
  `(-399,-235), (-827,-553), (-300,-237), (+202,+51)`; 827105
  `(+239,+305), (-140,-176), (+295,+381), (-220,-137)`; 191873
  `(-97,-22), (-53,-70), (-173,-173), (+171,-68)`. Runtime was 654.67s
  baseline and 709.63s candidate. The candidate did not generalize across the
  seeds and remains committed, unmerged.

The score-aim baseline itself illustrates run variance: its seed 827104 batch
scored 3,282 total points, 469 more than an earlier same-code, same-seed batch
at 2,813. This is evidence to treat small aggregate candidate differences
cautiously; paired per-world direction and replication matter. Records are in
`evaluations/score-aim-continuous/baseline/`,
`AutoDrive-score-aim-continuous/evaluations/score-aim-continuous/candidate/`,
`evaluations/pickup-braking-envelope/baseline/`, and
`AutoDrive-pickup-braking-envelope/evaluations/pickup-braking-envelope/candidate/`.

### Target hysteresis and clear-zone scoring trials (2026-10-11)

Both candidates were committed before evaluation and paired against runtime
behavior from `main` (`958ebd8`). They used the fixed 3-seed x 4-world protocol,
8,000 ticks, fuel physics, replan interval 20, perception interval 2, and no
teammate coordination. Each baseline/candidate pair ran sequentially on one GPU;
the candidate experiments used separate GPUs. These were end-to-end simulator
records, not dashboard replays; no dashboard replay or stack verification was
performed.

- **Robot-local target hysteresis (`14f1ce4`,
  `codex/target-hysteresis-current`)**: +165 acquisitions and +1,030 points;
  mean score 691.33→777.17. Per-seed fuel/score deltas: 827104 +234/+123,
  827105 −463/+153, 191873 +394/+754. Per-world fuel/score deltas: 827104
  `(+172,+94), (-42,-165), (+213,+297), (-109,-103)`; 827105
  `(-217,-165), (+135,+421), (-79,-10), (-302,-93)`; 191873
  `(-149,+137), (-100,+90), (+400,+405), (+243,+122)`. Runtime was
  671.46s baseline and 663.33s candidate. Strong aggregate points, but mixed
  seed and acquisition results leave it experimental and unmerged.
- **Score on entry to a clear active HUB zone (`4dc9486`,
  `codex/active-zone-score-commit`)**: −606 acquisitions and −638 points;
  mean score 760.00→707.83. Per-seed fuel/point deltas: 827104 +189/+231,
  827105 +258/−465, and 191873 −1,053/−404. Per-world fuel/point deltas:
  827104 `(-105,-140), (+56,+195), (+25,-135), (+213,+311)`; 827105
  `(-23,-122), (+179,-222), (+121,-96), (-19,-25)`; 191873
  `(-271,-107), (-186,-41), (-120,-167), (-476,-89)`. Runtime was 555.15s
  baseline and 605.65s candidate. The strategy change degraded the aggregate
  score and remains committed, unmerged.

Records are in `evaluations/target-hysteresis-current/baseline/`,
`AutoDrive-target-hysteresis-current/evaluations/target-hysteresis-current/candidate/`,
`evaluations/active-zone-score-commit/baseline/`, and
`AutoDrive-active-zone-score-commit/evaluations/active-zone-score-commit/candidate/`.

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

### Phase staggering plus target hysteresis bundle (2026-10-11)

Candidate `3aaf861` (`codex/phase-hysteresis-bundle`) combines raster phase
staggering with robot-local target hysteresis. It was evaluated against `main`
at `97df7ec` using seeds 827104, 827105, and 191873, four worlds per seed,
8,000 ticks, fuel physics, replan interval 20, perception interval 2, and no
teammate intent knowledge or coordination. This was a fresh end-to-end
simulator batch; dashboard replay and dashboard-stack verification were not
performed.

Across 12 worlds, the candidate gained 898 team fuel acquisitions and 233
team points. Mean score deltas by seed were +31.50, +75.75, and −49.00,
respectively. Per-world fuel/score deltas were 827104 `(-96,+47),
(+107,-180), (+250,+119), (+239,+140)`; 827105 `(+567,+429), (+333,-27),
(-24,+131), (-256,-230)`; and 191873 `(+156,+218), (+438,+330),
(-581,-545), (-235,-199)`. Runtime was 646.12 seconds baseline and 668.66
seconds candidate across the three seed batches. The per-world results are
mixed and seed 191873 regressed, so preserve this as an experimental alternate
strategy; do not merge it based on the aggregate alone.

Baseline records are in `evaluations/phase-hysteresis-bundle/baseline/` and
candidate records are in the corresponding candidate worktree at
`AutoDrive-phase-hysteresis-bundle/evaluations/phase-hysteresis-bundle/candidate/`.

### Pickup density plus target hysteresis bundle (2026-10-11)

Candidate `7419b50` (`codex/density-hysteresis-current-main`) combines dense
patch pickup targeting with robot-local target hysteresis. It was compared
against `main` at `97df7ec` on the same three seeds, four worlds per seed,
8,000 ticks, fuel physics, replan interval 20, perception interval 2, and no
teammate intent knowledge or coordination. The run was a fresh end-to-end
simulator batch; it did not use dashboard replay or dashboard-stack checks.

The candidate lost 873 team fuel acquisitions and 484 team points overall;
mean score fell from 733.08 to 692.75. Fuel/score deltas by seed were 827104
+272/+175, 827105 −490/−452, and 191873 −655/−207. Per-world fuel/score
deltas were 827104 `(+197,+163), (+162,+14), (-249,-152), (+162,+150)`;
827105 `(-321,-322), (+87,+19), (+438,+226), (-694,-375)`; and 191873
`(-621,-149), (+70,+132), (-89,-162), (-15,-28)`. Tick-loop runtime
improved from 751.43 to 691.43 seconds, but behavior quality regressed, so
retain the committed attempt without merging it.

Baseline records and per-seed comparison files are in
`evaluations/density-hysteresis-current-main/`; candidate records are in the
`AutoDrive-density-hysteresis` candidate worktree under the matching
evaluation directory.

### Dense-patch ranking plus target hysteresis bundle (2026-10-11)

Candidate `79b1e6d` (`codex/cluster-target-hysteresis-bundle`) combines dense
fuel-patch ranking with robot-local target hysteresis. It was paired against
the current behavior in `main` (`9afb152`, docs-only advances during the
batch) on seeds 827104, 827105, and 191873, four worlds per seed, 8,000 ticks,
fuel physics, replan interval 20, perception interval 2, and no teammate
intent knowledge or coordination. This was a fresh end-to-end simulator
evaluation; dashboard replay and dashboard-stack verification were not used.

Across 12 worlds, the candidate gained 1,264 team fuel acquisitions and 1,464
team points; mean score rose from 607.25 to 729.25. Mean score deltas by seed
were +222.00, −39.00, and +183.00. Per-world fuel/score deltas were 827104
`(+68,+234), (+240,+142), (-307,-88), (+874,+600)`; 827105
`(-54,-127), (-230,-177), (-47,-22), (+139,+170)`; and 191873
`(+604,+779), (-255,+27), (+36,-255), (+196,+181)`. Runtime was 635.30
seconds baseline and 668.06 seconds candidate across the three seed batches.
The aggregate is promising, but results vary by seed and mean remains below
800, so preserve this committed alternate strategy as experimental and unmerged.

Baseline records are in `evaluations/cluster-target-hysteresis/baseline/`;
candidate records are under the matching directory in the
`AutoDrive-cluster-target-hysteresis-bundle` worktree.

### No-track raster phases plus dense-patch ranking bundle (2026-10-11)

Candidate `25fb36d` (`codex/phase-density-current-main`) combines no-track
raster phase staggering with dense-patch fuel ranking, preserving the
obstacle-aware ranking on `main`. It was paired against `main` at `9afb152`
on seeds 827104, 827105, and 191873, four worlds per seed, 8,000 ticks, fuel
physics, replan interval 20, perception interval 2, and no teammate intent
knowledge or coordination. This was a fresh end-to-end simulator batch; no
dashboard replay or dashboard-stack verification was involved.

Across 12 worlds, this candidate lost 2,238 team fuel acquisitions and 633
team points. Mean score fell from 722.75 to 670.00. Fuel/score deltas by seed
were 827104 −1,224/−468, 827105 −253/−9, and 191873 −761/−156. Per-world
fuel/score deltas were 827104 `(-324,-21), (-85,-28), (-637,-389),
(-178,-30)`; 827105 `(+223,+26), (-90,+157), (-156,-15), (-230,-177)`; and
191873 `(+271,+388), (-346,-141), (-203,+38), (-483,-441)`. Runtime was
685.96 seconds baseline and 703.67 seconds candidate. This combination
regressed behavior and runtime; retain the committed attempt unmerged.

Baseline records are in `evaluations/phase-density-current-main/baseline/`;
candidate records are under the matching directory in the
`AutoDrive-phase-density` worktree.

### Dense-patch plus hysteresis unseen-seed holdout A (2026-10-11)

To test whether candidate `79b1e6d` generalizes beyond its development batch,
it was evaluated without tuning on previously unused seeds 64821, 352190, and
740813, four worlds per seed. Settings remained 8,000 ticks, fuel physics,
replan interval 20, perception interval 2, and no teammate intent knowledge
or coordination. Baseline and candidate were evaluated sequentially on GPU0;
the current-main revisions differ only by docs-only commits during the run.
This was a fresh end-to-end simulator holdout, with no dashboard replay or
dashboard-stack verification.

Across the 12 holdout worlds, fuel acquisitions changed by −12 and points by
+260; mean score rose from 685.92 to 707.58. Mean score increased on all
three seeds: 64821 +40.75, 352190 +13.75, and 740813 +10.50. Per-world
fuel/score deltas were 64821 `(+379,+270), (-368,-327), (-229,-255),
(+527,+475)`; 352190 `(0,+131), (0,+158), (+237,+209), (-363,-443)`; and
740813 `(+123,+89), (-397,-111), (-84,+12), (+163,+52)`. Runtime was
638.60 seconds baseline and 640.84 seconds candidate. The score effect is
modestly positive across seed means, but the fuel gate did not hold and world
results vary; keep the behavior experimental and unmerged.

Baseline records are in `evaluations/cluster-target-hysteresis-holdout/baseline/`;
candidate records are under the matching directory in the
`AutoDrive-cluster-target-hysteresis-bundle` worktree.

### Dense-patch plus hysteresis unseen-seed holdout B (2026-10-11)

The same frozen candidate `79b1e6d` was evaluated on a second preselected
holdout batch: seeds 925531, 184331, and 419039, four worlds per seed. Settings
were unchanged (8,000 ticks, fuel physics, replan interval 20, perception
interval 2, no teammate intent knowledge or coordination), with baseline and
candidate run sequentially on GPU1. This was a fresh end-to-end simulator
evaluation; no dashboard replay or dashboard-stack verification was used.

Across these 12 unseen worlds, the candidate gained 492 fuel acquisitions but
lost 159 points; mean score fell from 792.08 to 778.83. Per-seed fuel/score
deltas were 925531 −241/−421, 184331 +247/−32, and 419039 +486/+294. Per-world
fuel/score deltas were 925531 `(+13,-84), (-30,-41), (+65,+3), (-289,-299)`;
184331 `(+10,-193), (-64,-60), (+220,+121), (+81,+100)`; and 419039
`(-67,-178), (-23,-1), (+282,+371), (+294,+102)`. Runtime was 754.26
seconds baseline and 698.82 seconds candidate, excluding the candidate's
first-time compile overhead.

Across both independent holdout batches (24 worlds), the candidate's combined
delta was +480 fuel acquisitions and +101 points, a mean-score increase of
about 4.21 points per world. This is a small, mixed effect: holdout A was
nearly flat on fuel with a modest score gain, while holdout B gained fuel but
lost score. The fixed development batch's larger gain does not establish a
robust improvement or reach the 800-point target. Keep the branch experimental
and unmerged.

Baseline records are in `evaluations/cluster-target-hysteresis-holdout-b/baseline/`;
candidate records are under the matching directory in the
`AutoDrive-cluster-target-hysteresis-bundle` worktree.

### Phase, dense-patch, and target-hysteresis triple bundle (2026-10-11)

Candidate `7aa607f` (`codex/phase-density-hysteresis-bundle`) combines
no-track raster phase staggering, dense-patch ranking, and robot-local target
hysteresis. It was paired against current `main` on the original seeds 827104,
827105, and 191873, four worlds per seed, 8,000 ticks, fuel physics, replan
interval 20, perception interval 2, and no teammate intent knowledge or
coordination. This was a fresh end-to-end simulator batch; no dashboard replay
or dashboard-stack verification was used.

Across 12 worlds, the candidate gained 144 team fuel acquisitions and 59 team
points; mean score rose from 713.83 to 718.75. Mean score deltas by seed were
827104 +114.50, 827105 −106.50, and 191873 +6.75. Per-world fuel/score deltas
were 827104 `(-44,-31), (+384,+568), (-201,-132), (+311,+53)`; 827105
`(+217,+101), (+85,-30), (-219,-189), (-307,-308)`; and 191873
`(-185,-63), (+195,+120), (+66,+66), (-158,-96)`. Runtime was 639.56
seconds baseline and 726.37 seconds candidate. This is a small, mixed result,
with a large regression on seed 827105 and increased runtime; preserve the
branch as an alternate experiment and leave it unmerged.

Baseline records are in `evaluations/phase-density-hysteresis/baseline/`;
candidate records are in the matching directory in the
`AutoDrive-phase-density-hysteresis-bundle` worktree.

### Siloed lane search (2026-10-11)

Candidate `c625676` (`codex/siloed-lane-search-holdout`) changes the existing
no-track search fallback to a static field lane derived from the acting
robot's team ID and slot. The path does not read teammate intent or teammate
state. It was paired against current `main` on the original seeds 827104,
827105, and 191873, four worlds per seed, 8,000 ticks, fuel physics, replan
interval 20, perception interval 2, and `--no-teammate-coordination`. This was
a fresh end-to-end simulator batch, not a dashboard replay.

Across 12 worlds, the candidate gained 1,183 team fuel acquisitions and 945
team points; mean score rose from 659.67 to 738.42. Fuel/score totals by seed
were 827104 −52/+53, 827105 +417/+487, and 191873 +818/+405. Per-world
fuel/score deltas were 827104 `(+292,+139), (-301,-256), (-22,+75),
(-21,+95)`; 827105 `(+170,+245), (+234,+127), (-210,-95), (+223,+210)`; and
191873 `(-136,-28), (+876,+412), (-112,-120), (+190,+141)`. Runtime was
693.08 seconds baseline and 638.74 seconds candidate. The gain is promising
but concentrated in two seeds, and the mean remains below 800; keep the
committed behavior experimental until it passes unseen-seed evaluation.

The baseline's first two JSON records identify `d520ed3`, and its last record
identifies `f8d5872`; the intervening change was docs-only. Candidate records
are in `AutoDrive-siloed-lane-search/evaluations/siloed-lane-search-holdout/candidate/`;
baseline records are in `evaluations/siloed-lane-search-holdout/baseline/`.

### Siloed lane search unseen-seed holdout C (2026-10-11)

The frozen candidate `c625676` was evaluated on seeds 390671, 817931, and
260989, four worlds per seed. Settings remained 8,000 ticks, fuel physics,
replan interval 20, perception interval 2, and no teammate intent knowledge or
coordination. This was a fresh end-to-end simulator evaluation, not a dashboard
replay.

Across 12 unseen worlds, fuel acquisitions increased by 166 while points
decreased by 200; mean score fell from 745.00 to 728.33. Mean score deltas by
seed were 390671 −139.25, 817931 +68.50, and 260989 +20.75. Per-world
fuel/score deltas were 390671 `(-108,-46), (+2,+47), (-244,-398), (-163,-160)`;
817931 `(-248,-118), (-64,-122), (+586,+474), (-24,+40)`; and 260989
`(+209,+36), (-153,-186), (+11,+26), (+362,+207)`. Runtime was 612.96 seconds
baseline and 620.30 seconds candidate.

The positive fuel total did not translate to a score gain and the seed effects
vary substantially. Preserve the committed candidate branch as an alternate
attempt, but do not merge it or treat this as a generalized improvement.

Baseline records are in `evaluations/siloed-lane-search-holdout-c/baseline/`;
candidate records are under the matching directory in the
`AutoDrive-siloed-lane-search` worktree.

### Siloed lane search unseen-seed holdout B (2026-10-11)

The same frozen candidate `c625676` was evaluated on seeds 579239, 420221,
and 885113, four worlds per seed. Settings remained 8,000 ticks, fuel physics,
replan interval 20, perception interval 2, and no teammate intent knowledge or
coordination. This was a fresh end-to-end simulator evaluation, not a dashboard
replay.

Across 12 unseen worlds, fuel acquisitions increased by 599 and points by 850;
mean score rose from 788.08 to 858.92. Fuel/score deltas by seed were 579239
+622/+289, 420221 −180/+288, and 885113 +157/+273. Per-world fuel/score deltas
were 579239 `(+440,+253), (+309,+224), (+34,-33), (-161,-155)`; 420221
`(-12,+120), (-450,-205), (+103,+238), (+179,+135)`; and 885113
`(-68,+15), (+300,+278), (-162,-57), (+87,+37)`. Runtime totals were 611.74
seconds baseline and 642.63 seconds candidate.

This holdout was positive on aggregate fuel and score, though fuel improved in
7/12 worlds and the per-seed fuel effect was mixed. Combined with independent
holdout C, the two unseen batches (24 worlds) yielded +765 fuel and +650 points
(+27.08 mean score per world). This supports keeping the committed lane-search
branch as a promising experimental alternative, but not merging it: holdout C
lost 200 points, and the combined evidence remains limited to two batches.

Baseline records are in `evaluations/siloed-lane-search-holdout-b/baseline/`;
candidate records are under the matching directory in the
`AutoDrive-siloed-lane-search` worktree.

### Siloed lane search unseen-seed holdout D (2026-10-11)

The frozen candidate `c625676` was compared with current-main baseline
`1900abe` on seeds 305411, 693217, and 158329, four worlds per seed. Settings
were 8,000 ticks, fuel physics, replan interval 20, perception interval 2, and
no teammate intent knowledge or coordination. This was a fresh end-to-end
simulator evaluation, not a dashboard replay.

Across 12 worlds, fuel acquisitions increased by 209 and points by 241; mean
score rose from 706.42 to 726.50. Fuel/score deltas by seed were 305411
−70/−191, 693217 +195/+262, and 158329 +84/+170. Per-world fuel/score deltas
were 305411 `(+11,+85), (-158,-212), (+507,+339), (-430,-403)`; 693217
`(-184,-71), (+113,+117), (+106,+320), (+160,-104)`; and 158329
`(-463,-218), (+223,-50), (+236,+300), (+88,+138)`. Runtime totals were
681.01 seconds baseline and 631.20 seconds candidate.

The aggregate was positive, but the first seed regressed and the mean remained
below 800. Keep the already committed lane-search branch as an isolated
experimental option; this batch does not justify merging it. Across all three
unseen lane-search holds B, C, and D (36 worlds), total change is +974 fuel
and +891 points, with score losses in holdout C and a combined mean-score
gain of 24.75 points per world.

Baseline records are in `evaluations/siloed-lane-search-holdout-d/baseline/`;
candidate records are in the matching main evaluation directory and the
`AutoDrive-siloed-lane-search` worktree. Both record sets identify their
respective revisions; candidate and baseline generated HIP sources matched by
hash. Dashboard replay and stack checks were not performed.

### Siloed lane search independent holdout H (2026-10-11)

The frozen candidate `c625676` was paired against current-main revision
`60b2aaa` on seeds 615379, 704167, and 992791, four worlds per seed. Both
record sets used 8,000 ticks, fuel physics, replan interval 20, perception
interval 2, and no teammate intent or coordination. The candidate evaluator
used `PYTHONPATH` pinned to its worktree and verified its imported package and
behavior module paths before launch. Candidate ran on `cuda:1`, paired with
the matching device-specific main baseline; scenario IDs match world by
world.

Across 12 worlds, raw fuel acquisitions fell by 30 while team-0 scored fuel
and score differential each rose by 80. Per-seed acquisition/scored-fuel/point
deltas were 615379 +300/+280/+280, 704167 +125/−41/−41, and 992791
−455/−159/−159. Acquisitions improved in 5/12 worlds and scored fuel in 4/12.
This is mixed evidence, not a general collection gain; keep lane search as an
isolated, unmerged strategy. Its result record is committed on the strategy
branch at `33d7bb5`, under
`AutoDrive-siloed-lane-search/docs/siloed-lane-search-holdout-H.md`. No
dashboard replay was generated or checked.

### No-track raster phase staggering unseen-seed holdout D (2026-10-11)

Frozen candidate `6d9b709` was compared with current-main baseline `1900abe`
on seeds 740021, 268933, and 913507, four worlds per seed. Settings were
8,000 ticks, fuel physics, replan interval 20, perception interval 2, and no
teammate intent knowledge or coordination. This was a fresh end-to-end
simulator evaluation, not a dashboard replay.

Across 12 unseen worlds, fuel acquisitions increased by 871 and points by 799;
mean score rose from 696.67 to 763.25. Fuel/score deltas by seed were 740021
+262/+175, 268933 +24/+90, and 913507 +585/+534. Per-world fuel/score deltas
were 740021 `(+112,+26), (+158,+40), (+92,+104), (-100,+5)`; 268933
`(+242,+137), (-372,-356), (+131,+150), (+23,+159)`; and 913507
`(-254,-158), (+228,+285), (+547,+384), (+64,+23)`. Runtime totals were
611.38 seconds baseline and 629.85 seconds candidate. Fuel improved in 9/12
worlds and score in 10/12.

This is a solid positive batch, though its aggregate mean remains below 800 and
its elapsed time was about 3% higher. Preserve `6d9b709` as a separate,
unmerged behavior option; continue testing generalization before adoption.

Baseline records are in `evaluations/no-track-phase-holdout-d/baseline/`;
candidate records are in the matching main evaluation directory and the
`AutoDrive-no-track-raster-phase-recheck` worktree. Candidate and baseline
generated HIP sources matched by hash. Dashboard replay and stack checks were
not performed.

### Fresh phase and lane strategy holdout E (2026-10-11)

These are separate candidate branches evaluated independently against the same
current-main revision (`7280584`); they were not combined or enabled
simultaneously. Each evaluation used three fresh seeds and four paired world
slots per seed, 8,000 ticks, fuel physics, replan interval 20, perception
interval 2, and disabled teammate intent knowledge and coordination. The saved
records identify baseline revision `7280584`, phase candidate `6d9b709`, and
lane-search candidate `c625676`. Candidate records were copied to the main
evaluation directories and verified against their originals.

- **No-track raster phase staggering**, seeds 684293, 217531, and 907163:
  +2,190 fuel acquisitions and +1,965 points over 12 worlds; mean score
  658.50→822.25. Fuel and score both improved in 8/12 worlds. Seed fuel/score
  deltas were 684293 +702/+684, 217531 +683/+395, and 907163 +805/+886.
  Per-world fuel/score deltas were 684293 `(-141,-13), (+502,+735),
  (-69,-81), (+31,+43)`; 217531 `(-115,+41), (+308,+258), (+74,-75),
  (+416,+171)`; 907163 `(+334,+309), (-104,-62), (+339,+414), (+236,+225)`.
  Summed evaluator time was 621.74 seconds baseline and 635.11 seconds
  candidate (+2.2%). This was a fresh end-to-end simulator batch, not a
  dashboard replay.
- **Siloed lane search**, seeds 326741, 581023, and 947351: −1,548 fuel
  acquisitions and −826 points over 12 worlds; mean score 780.83→712.00.
  Seed fuel/score deltas were 326741 −1,146/−825, 581023 +88/−21, and 947351
  −490/+20. Per-world fuel/score deltas were 326741 `(-177,-58),
  (-377,-186), (-400,-242), (-192,-339)`; 581023 `(-110,-107), (+81,+112),
  (-81,-116), (+198,+90)`; 947351 `(-47,+101), (+15,+73), (-9,+151),
  (-449,-305)`. Summed evaluator time was 641.42 seconds baseline and
  634.50 seconds candidate. This was a fresh end-to-end simulator batch, not a
  dashboard replay.

Keep the two committed behaviors as separate selectable options:
`6d9b709` on `codex/no-track-raster-phase-recheck` has a positive holdout E,
while `c625676` on `codex/siloed-lane-search-holdout` regressed on holdout E.
Neither candidate was merged. Records are in
`evaluations/no-track-phase-holdout-e/{baseline,candidate}/` and
`evaluations/siloed-lane-search-holdout-e/{baseline,candidate}/`. No dashboard
replay was generated, so dashboard stack status was not part of this
validation.

### Separate phase-stagger and target-hysteresis holdout F (2026-10-11)

The frozen phase-stagger candidate `6d9b709` and target-hysteresis candidate
`14f1ce4` were each evaluated separately against baseline `d95fb1a` on the
same fresh seeds: 416827, 730159, and 982451. Each used four world slots,
8,000 ticks, fuel physics, replan interval 20, perception interval 2, and no
teammate intent or coordination. Phase baseline/candidate ran sequentially on
`cuda:1`; target-hysteresis baseline/candidate ran sequentially on `cuda:0`.
Generated HIP source hashes matched across the baseline and candidate
worktrees. Their raw baseline outcomes differed between devices, so compare
each candidate only with its paired baseline on the same device; do not compare
absolute scores across the two arms.

- **Phase staggering:** −104 fuel and −128 points over 12 worlds; mean score
  778.67→768.00. Fuel/score deltas by seed were 416827 +249/+221, 730159
  −230/−283, and 982451 −123/−66. Per-world fuel/score deltas were 416827
  `(+31,-38), (-487,-237), (+130,+218), (+575,+278)`; 730159
  `(-445,-234), (-573,-633), (+508,+364), (+280,+220)`; 982451
  `(+89,+6), (-163,-172), (-165,+95), (+116,+5)`. Both fuel and score
  improved in 7/12 worlds. Summed evaluator time was 630.42 seconds baseline
  and 657.72 seconds candidate. This is a fresh end-to-end simulator batch,
  not a dashboard replay.
- **Robot-local target hysteresis:** +759 fuel and +1,167 points over 12
  worlds; mean score 702.08→799.33. Fuel/score deltas by seed were 416827
  +693/+695, 730159 −125/+221, and 982451 +191/+251. Per-world fuel/score
  deltas were 416827 `(-32,+52), (+568,+367), (+60,+135), (+97,+141)`;
  730159 `(-404,-103), (+116,+41), (-106,-17), (+269,+300)`; 982451
  `(+467,+165), (+15,+81), (+71,+191), (-362,-186)`. Summed evaluator time
  was 621.56 seconds baseline and 665.38 seconds candidate. This is a fresh
  end-to-end simulator batch, not a dashboard replay.

These are alternative behavior branches; no candidate was combined with the
other or merged. Target hysteresis gained aggregate fuel on this holdout, while
phase staggering regressed. Both remain committed and selectable, with more
unseen-seed evidence needed before adoption. Records are preserved in
`evaluations/phase-compat-holdout-f/phase-only/{baseline,candidate}/` and
`evaluations/target-hysteresis-holdout-f/{baseline,candidate}/`. No dashboard
replay was generated, so dashboard stack status was not part of validation.

### Target hysteresis and phase-hysteresis bundle holdout G (2026-10-11)

The frozen target-hysteresis candidate `14f1ce4` and phase-plus-hysteresis
bundle `3aaf861` were evaluated separately against current-main behavior
baseline `75df5b2` on fresh seeds 209317, 637951, and 841273. Each used four
world slots, 8,000 ticks, fuel physics, replan interval 20, perception
interval 2, and no teammate intent or coordination. Target-hysteresis runs
used `cuda:0`; the bundle runs used `cuda:1`, with baseline and candidate
sequential on each assigned device. Baseline score outcomes differed between
devices for these same seed labels, so do not compare absolute scores across
the two candidate arms. Candidate JSON copies were checked against the
worktree copies by SHA-256.

- **Target hysteresis alone:** −405 fuel and −688 points over 12 worlds; mean
  score 753.08→695.75. Seed fuel/score deltas were 209317 −152/−87, 637951
  +11/−4, and 841273 −264/−597. Per-world fuel/score deltas were 209317
  `(+150,+125), (+71,+127), (-351,-252), (-22,-87)`; 637951
  `(+88,-12), (-53,-83), (+50,+53), (-74,+38)`; 841273
  `(-318,-351), (-115,-395), (+224,+219), (-55,-70)`. Summed evaluator time
  was 643.91 seconds baseline and 635.53 seconds candidate.
- **Phase staggering plus target hysteresis:** +31 fuel and +469 points over
  12 worlds; mean score 720.33→759.42. Fuel/score deltas by seed were 209317
  −559/−222, 637951 +81/−174, and 841273 +509/+865. Per-world fuel/score
  deltas were 209317 `(-308,-348), (+289,+318), (-256,-76), (-284,-116)`;
  637951 `(-70,-53), (-7,-275), (+155,+133), (+3,+21)`; 841273
  `(+134,+183), (+245,+248), (+295,+348), (-165,+86)`. Fuel improved in
  6/12 worlds and score in 7/12. Summed evaluator time was 641.41 seconds
  baseline and 627.32 seconds candidate.

Holdout G weakened the target-hysteresis-only result after its positive F
batch. The combined branch gained only 31 fuel and remained far below 800 mean
score, so treat it as an experimental alternative rather than an established
improvement. Neither branch was merged. Records are in
`evaluations/target-hysteresis-holdout-g/{baseline,candidate}/` and
`evaluations/phase-hysteresis-holdout-g/{baseline,candidate}/`. These were
fresh end-to-end simulator batches; no dashboard replay or stack check was
performed.

### FMS local action-window independent holdout H (2026-10-11)

The isolated FMS candidate `e016b22` on `codex/fms-local-action-window` was
paired against current-main revision `60b2aaa` on seeds 615379, 704167, and
992791, four worlds per seed. Candidate and baseline both used 8,000 ticks,
fuel physics, replan interval 20, perception interval 2, and no teammate
intent or coordination. Candidate ran on `cuda:0` with `PYTHONPATH` pinned to
its worktree; package and behavior-module import paths were verified. Its
records match the main baseline scenarios world by world.

Across 12 worlds, offense fuel acquisitions increased by 1,001 and points by
557; mean score rose from 634.75 to 681.17. Per-seed fuel/point deltas were
615379 +768/+271, 704167 +156/−6, and 992791 +77/+292. Fuel improved in 8/12
worlds and points in 7/12. Aggregate fuel was positive on all three seeds, but
the mean remained below 800 and world outcomes were mixed. Keep the behavior
as a separate unmerged strategy; this result does not validate combining it
with phase staggering or lane search. Provenance, per-world deltas, and JSON
records are committed on that branch at `cadb8bc` in
`AutoDrive-fms-local-action-window/evaluations/fms-local-action-window-holdout-H-gpu0/`.
No dashboard replay was generated or checked.

### Phase and FMS strategy independent holdout J (2026-10-11)

The frozen phase-stagger (`6d9b709`) and FMS local action-window (`e016b22`)
strategies were each evaluated alone against main revision `737c886` on fresh
seeds 153719, 246817, and 390527, four worlds per seed. Both arms used 8,000
ticks, fuel physics, replan 20, perception 2, and disabled teammate intent and
coordination. Phase ran on `cuda:0`; FMS ran on `cuda:1`; each baseline was
paired on the same device. Candidate worktree imports were verified before
launch, and the comparison script confirmed identical scenario IDs and
settings.

- **Phase staggering alone:** −873 fuel acquisitions and −329 points over 12
  worlds; mean score 748.92→721.50. Seed fuel/score deltas: 153719 −471/−168,
  246817 +154/+112, 390527 −556/−273. Fuel improved in 5/12 worlds and score
  in 4/12. This fresh holdout regresses the aggregate.
- **FMS action-window filtering alone:** +364 fuel acquisitions and +10 points
  over 12 worlds; mean score 757.83→758.67. Seed fuel/score deltas: 153719
  +687/+414, 246817 +99/+110, 390527 −422/−514. Fuel improved in 7/12 worlds
  and score in 5/12. The small aggregate fuel gain is seed-sensitive and did
  not translate to a material score gain.

These independent runs show why the strategy branches remain separate. Neither
candidate exceeds 800 mean score or establishes reliable generalization; do not
merge either behavior based on these results. In particular, standalone
results do not demonstrate that strategies combine well. Full batch JSON,
paired comparisons, logs, and per-strategy reports are committed on their
respective branches: phase `142af44` on
`codex/no-track-raster-phase-recheck`; FMS `ecd50ce` on
`codex/fms-local-action-window`. This was end-to-end simulator validation, not a
dashboard replay; dashboard stack status was not checked.

### Pre-registration: two local score-commit candidates holdout K (2026-10-11)

Before either candidate is evaluated, freeze this paired comparison for seeds
301927, 570381, and 938617, four vector worlds per seed. Candidate A is the
active-FMS pickup/return gate (`87e425f`, `codex/local-active-window-pickup-gate`);
candidate B is the end-of-match loaded scoring gate (`0641100`,
`codex/robot-local-endgame-score-commit`). Evaluate each candidate alone against
the same current-main behavior on a dedicated device: A on `cuda:0`, B on
`cuda:1`, with baseline and candidate sequential on that device. Use 8,000
ticks, fuel physics, replan interval 20, perception interval 2, and disabled
teammate intent/coordination. Do not combine candidates or tune from these
seeds. Record all per-world fuel and score outcomes and keep each attempt on
its own branch regardless of result. The three existing uncommitted HIP
runtime source edits in the main checkout are copied byte-for-byte to each
candidate worktree before evaluation so native simulator code matches. The
pre-registered seeds were checked against this comparison document and have
not been evaluated in this experiment.

Holdout K completed on both devices with the required settings and candidate
imports verified from their worktrees:

- **Local active-window pickup/return gate** (`87e425f`, fixed at `cb51892`):
  +255 fuel and +386 points over 12 worlds; mean score 641.92→674.08. Seed
  fuel/score deltas: 301927 +809/+709, 570381 −261/+18, 938617 −293/−341.
  Fuel and score each improved in 5/12 worlds. The first launch failed before
  producing candidate rows due to a tensor index error; that failure log is
  preserved and the corrected full batch passed. Since the gain is driven by
  one seed and the other two lose fuel, retain it only as an experimental,
  separate strategy.
- **Robot-local endgame score commitment** (`0641100`): +198 fuel and +415
  points over 12 worlds; mean score 632.67→667.25. Seed fuel/score deltas:
  301927 +80/+114, 570381 −668/−242, 938617 +786/+543. Fuel improved in
  5/12 worlds and score in 7/12. The direction is seed-sensitive; retain it
  only as an experimental, separate strategy.

The candidates remain committed and unmerged and have not been combined with
each other. Their complete baseline/candidate records, paired comparisons, and
logs are committed on their branches: active-window `1169cac` on
`codex/local-active-window-pickup-gate`; endgame `d85d9c6` on
`codex/robot-local-endgame-score-commit`. Neither reaches the >800 mean-score
target or establishes broad generalization. These were fresh simulator batches,
not dashboard replays; dashboard stack status was not checked.

### Independent local score-gate replication L (2026-10-11)

The same two frozen candidates were evaluated on fresh seeds 413807, 725961,
and 884317, four worlds per seed. The active-window candidate ran on `cuda:0`
and endgame score commitment on `cuda:1`, each paired sequentially against the
same main behavior on its device. Settings and no-coordination policy matched
Holdout K; candidate imports and common HIP source bytes were verified.

- **Active-window pickup/return gate:** +20 fuel and +91 points over 12 worlds;
  mean score 848.42→856.00. Seed fuel/score deltas: 413807 +28/−55, 725961
  −29/+134, 884317 +21/+12. Fuel improved in 6/12 worlds and score in 5/12.
  The aggregate fuel change is near zero. Across K and L, it gained 275 fuel
  and 477 points in 24 worlds; mean score was 745.17→765.04. Keep it isolated
  as an experimental alternative.
- **Endgame score commitment:** +262 fuel and +735 points over 12 worlds;
  mean score 806.58→867.83. Seed fuel/score deltas: 413807 +223/+257, 725961
  +128/+207, 884317 −89/+271. Score improved on every seed and in 8/12
  worlds; fuel improved in 9/12. Holdout K was weaker (+198/+415, mean
  632.67→667.25). Across K and L it gained 460 fuel and 1,150 points in 24
  worlds; combined mean score was 719.63→767.54. L exceeds the >800 target,
  but the two-batch combined mean does not yet; preserve as a promising
  separate strategy pending another unseen-seed replication.

Full L records are committed on the strategy branches: active-window `04d9ab9`
and endgame `a70dcbc`. Neither branch is merged. The active-window worktree's
first L launch failed before simulation because it had no local virtualenv; the
failure is logged and the retry used the shared interpreter. No dashboard
replay or dashboard stack check was performed.

### Pre-registration: phase strategy and endgame-gate holdout M (2026-10-11)

Evaluate two unmerged strategies separately on new seeds 521709, 736481, and
991027, four worlds per seed: frozen phase staggering (`6d9b709`) on `cuda:0`
and frozen endgame score commitment (`0641100`) on `cuda:1`. Pair each with
current-main behavior (pre-run snapshot `3320480`, with no behavior-code
changes since `c8899b1`) sequentially on its device. Use 8,000 ticks,
fuel physics, replan 20, perception 2, and no teammate intent/coordination.
These seeds were checked against saved batch records before this entry was
committed. The candidates remain separate; no combined run or seed-specific
tuning is planned. The three native HIP source files are byte-identical across
the baseline and candidate worktrees.

#### Endgame candidate result

The endgame score-gate run completed with paired seed deltas of −145 fuel and
−69 points over 12 worlds; mean score fell from 755.67 to 749.92. Per-seed
fuel/score deltas were +157/−122, −334/−166, and +32/+219. Fuel improved in
7/12 worlds and score in 8/12; two seed aggregates gained fuel, while only
one gained score. This is mixed evidence and does not support promotion. Keep
the behavior as a separate selectable branch, with its full records committed
as `e9200a1` on `codex/robot-local-endgame-score-commit`; do not combine it
with another strategy without a dedicated evaluation.

The phase staggering candidate also completed on the same paired batch. It
lost 34 fuel acquisitions and 567 points overall; mean score fell from 709.00
to 661.75. Per-seed fuel/score deltas were +768/+466, −511/−632, and
−291/−401. Fuel improved in 6/12 worlds and score in 3/12. This is a clear
aggregate regression and poor cross-seed generalization. Preserve the candidate
as a separate selectable branch with full records committed as `0147e73` on
`codex/no-track-raster-phase-recheck`; do not combine it with another strategy
without a dedicated evaluation.

Both candidates were evaluated individually against the same baseline, never
simultaneously. Their full records, per-seed comparisons, and logs are
committed on their respective branches. The simulator runs used no teammate
intent knowledge or coordination; no dashboard replay was generated or
checked.

### Pre-registration: local score-gate replication L (2026-10-11)

To check generalization without combining or retuning, replicate the two
Holdout K robot-local gates on new seeds 413807, 725961, and 884317, four
worlds per seed. Evaluate the local active-window pickup/return gate at
`1169cac` on `cuda:0` and endgame score commitment at `d85d9c6` on `cuda:1`,
each paired sequentially against current-main behavior (`030663b`) on the same
device. Use the same 8,000-tick, fuel-physics, replan-20, perception-2 setup
and no teammate intent/coordination. These seeds were checked against saved
batch records before this entry was committed. Keep each strategy isolated and
retain every outcome; no tuning or combination will be based on this batch.

### Pre-registration: FMS-window and endgame-gate holdout N (2026-10-11)

Replicate two previously fuel-positive standalone behaviors on fresh seeds
852739, 467821, and 912463, four worlds per seed: FMS local action-window
filtering (`e016b22`, `codex/fms-local-action-window`) on `cuda:0`, and
robot-local endgame score commitment (`0641100`,
`codex/robot-local-endgame-score-commit`) on `cuda:1`. Pair each candidate
separately with the same current-main behavior snapshot (`b8f53e8`, docs-only
after behavior snapshot `3320480`) on its assigned device. Use 8,000 ticks,
fuel physics, replan interval 20, perception interval 2, and no teammate intent
or coordination. Exact-seed searches across the project docs and JSON records
found no existing occurrence of these seeds. Keep the candidates separate,
freeze their implementations, and retain results whether positive or negative;
this is a generalization holdout, not a tuning batch.
