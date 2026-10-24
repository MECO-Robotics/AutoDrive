# Phase 5 dashboard queue reconciliation

**Scope:** show the current supervisor queue attempt when its run ID is newer
than an older per-track manifest, without borrowing evidence from that old run.
This is a dashboard projection correction in the isolated candidate checkout.

The planner traced a stale dashboard status to a current C2 queue lane whose
run ID differed from the older aborted C2 `run.json`. The supervisor queue
showed a new queued attempt, then a running attempt, while the dashboard kept
showing the prior aborted manifest. The planner scoped a dashboard-only
projection fix; no supervisor, trainer, checkpoint, protocol, or live main
source changes were in scope.

Implementation was committed on `codex/meco-production-input-parity-integration`
in these commits:

- `f191cb07e66ba421ba427925586a10ac7baafd51` — project fresh queue attempts
  over older manifests;
- `0416c4d547798ad193c40b49871927003a86c76d` — keep unmanifested attempts
  unconfirmed and separate previous-run metadata;
- `94e8d928e86fff218297ad71e27980212afb9176` — validate queue run IDs/statuses
  and use the safe projection for queue-only fallback;
- `00cdd083bc7437ab95e413d48f36f51ea50b2470` — scope process ID evidence to
  fresh queue probes and report it separately from heartbeat evidence.

For a fresh queue lane with no matching manifest, only a `queued` lane is shown
as queued. Running, stalled, terminal, and unknown states remain unconfirmed
until a matching run manifest exists. The projection does not attach the old
run's checkpoint, metrics, heartbeat, configuration, source identity, or error;
it retains only previous run ID, directory, and states as history. Empty run IDs
and unknown lane statuses are rejected. Queue-probed process evidence is kept
separate from manifest heartbeat evidence, and stale queue-only views do not
expose a process ID. Queue-only rows make no source or protocol binding claim.

The independent reviewer approved exact final commit
`00cdd083bc7437ab95e413d48f36f51ea50b2470` after the implementer addressed
three successive review findings. `git diff --check` passed. No tests, simulator
evaluation, training, qualification, or hardware check were run for this
tranche.

**Limits:** this improves dashboard status accuracy only. It provides no evidence
about training behavior or qualification. The Python source change remains in
the dashboard candidate checkout and is not integrated into live main, whose
source identity is bound to active checkpoints.
