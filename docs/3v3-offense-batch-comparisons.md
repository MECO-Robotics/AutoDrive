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
- **Promising but unmerged:** staggering no-track raster phases has two
  independent aggregate-positive replications (+741 acquisitions/+483 points
  and +796/+923). Seed and world results remain mixed, and the mean scores
  were 748.92 and 777.00, so it has not established a robust >800 result.
- **New speed-lookahead trial:** nominally gained 16 acquisitions and 184
  points, but fuel-positive evidence was mixed across seeds and close to the
  observed run variance. Keep it experimental and unmerged.
- **New target-hysteresis trial:** gained 165 acquisitions and 1,030 points,
  but one seed lost 463 acquisitions and mean score remained 777.17. Retain the
  branch for alternate strategy work; it has not established a robust >800 mean.
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
| Stagger no-track raster phases by robot slot | Historical comparison against an older raster baseline: +508 acquisitions, +15 points. Initial recheck: +58 acquisitions, +413 points across 12 worlds; seed fuel/score deltas 827104 +217/+370, 827105 +150/+161, 191873 −309/−118; fuel improved in 5/12 worlds. Two independent replications against current behavior baseline: A +741 fuel/+483 points (8/12 worlds fuel-positive, 8/12 score-positive; seed deltas +748/+551, −586/−648, +579/+580); B +796 fuel/+923 points (7/12 fuel-positive, 7/12 score-positive; seed deltas +232/+398, −132/−46, +696/+571). Replication mean scores: A 708.67→748.92, B 700.08→777.00. | Committed as `6d9b709` on `codex/no-track-raster-phase-recheck`; not merged. Both replications are aggregate-positive, but seed/world variance remains high and neither mean reached 800. Keep experimental pending broader robustness evidence. |
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
| Retain a robot's selected fuel target while it remains competitive, scaling the margin by its own targetward motion (`14f1ce4`, `codex/target-hysteresis-current`) | Current-main paired batch: +165 acquisitions and +1,030 points; mean score 691.33→777.17. Seed fuel/score deltas: 827104 +234/+123, 827105 −463/+153, 191873 +394/+754. Fuel improved in 5/12 worlds, score in 7/12. Runtime was 671.46s baseline and 663.33s candidate. | Promising aggregate score gain but mixed by seed, and mean remained below 800. Keep committed for alternate strategy use, experimental and unmerged. Records are in `evaluations/target-hysteresis-current/baseline/` and `AutoDrive-target-hysteresis-current/evaluations/target-hysteresis-current/candidate/`. |
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
