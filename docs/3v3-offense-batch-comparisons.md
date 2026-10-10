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
