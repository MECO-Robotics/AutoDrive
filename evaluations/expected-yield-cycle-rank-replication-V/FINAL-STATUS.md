# Expected-yield cycle-rank replication V: closeout

Date: 2026-10-11 UTC
Disposition at the initial stop: no completed rollout existed then. The study later resumed; see `INTERIM-RESULTS-2-SEEDS.md` for the current paired evidence.

## What was attempted

The preregistered six-seed replication compared the frozen robot-local expected-yield-per-cycle candidate (`a580691`) with baseline (`d434694`). Settings, seed order, evaluator hashes, and analysis were recorded in `PREREGISTRATION.md` before launch. Two agents started one baseline batch each:

| GPU | Seed | Start (UTC) | Stop (UTC) | Result |
| --- | ---: | --- | --- | --- |
| 0 | 339752 | 23:11:13 | 23:12:36 | SIGINT (130) during first-run HIP extension compilation |
| 1 | 973688 | 23:11:15 | 23:12:36 | SIGINT (130) during first-run HIP extension compilation |

Both first launches were stopped to honor the quick closeout and produced no batch result. They were later resumed with new logs and the originally empty preregistered result paths. The first two completed paired seed clusters are recorded in `INTERIM-RESULTS-2-SEEDS.md`; the full six-seed campaign remains open.

The full invocations and interruption output are preserved in `logs/`. Partial HIP build caches remain outside this records checkout at `/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive-cycle-rank-replication-artifacts/cache/`.

## Validation and workspace state

- Simulator behavior tests: none completed. The two processes stopped before an end-to-end match result existed.
- Dashboard/replay: none generated; the dashboard stack was not checked.
- GPU release at the initial stop: both devices reported 0% busy and about 8 MB VRAM used, with no evaluator/compiler process remaining. The later resumed campaign is using the devices as recorded in its run logs.
- Source worktrees were untouched. Main was not committed to or merged. The attempt record belongs on `codex/expected-yield-cycle-rank-replication-records`.

## Context for the optimization goal

The broader record remains in `docs/3v3-offense-batch-comparisons.md` and in `docs/3v3-offense-quick-closeout-2026-10-11.md` at commit `2e33454` on `codex/offense-goal-wrap-up`. A single 24-world top-eight run had median 857, but its fresh replication median was 740.5 (mean 777.92; 11/24 worlds above 800). The mainline obstacle-aware batch averaged 804.08, with 5/12 worlds above 800. Those results explain why one 800+ run did not establish dependable fresh-seed performance. The new cycle-rank replication did not resolve that question.

No experiment from this closeout is recommended for merge. The candidate and all other behavior experiments remain isolated until a complete, repeatable evaluation supports promotion.
