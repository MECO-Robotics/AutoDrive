# MECO dashboard data root

The dashboard can serve its own UI and campaign workflow ledger while reading
live MECO run state from another project checkout. This is useful when the
training checkout must remain untouched while a reviewed dashboard checkout is
used for inspection.

Start the dashboard from this checkout with:

```sh
.venv/bin/python -m frc_defense.dashboard \
  --run-dir '/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive-meco-final-plan-training/checkpoints/meco-five' \
  --host 127.0.0.1 --port 8771 \
  --meco-data-root '/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive-meco-final-plan-training'
```

Open `/meco`. The `/api/meco-status` payload reads queue state, run manifests,
heartbeats, statistics, checkpoints, evaluation records, and qualification
status from the selected data root. It reads `docs/MECO_CAMPAIGN_PASSES.json`
from the dashboard code checkout, so the planner/implementer/reviewer cards and
Phase 4 factor cards describe the code being served.

`--meco-data-root` must resolve to an existing directory. MECO checkpoint and
evaluation roots are fixed to its `checkpoints/meco-five` and
`evaluations/meco-five` children. Every nested status, run, heartbeat, stats,
evaluator, report, and qualification JSON path is resolved and checked against
that root before it is read, so symlinked status inputs outside the selected
data root are rejected. The campaign ledger remains checkout-local.
The default, when the option is omitted, keeps both live data and campaign
ledger checkout-local. An explicit `--curriculum-root` may point inside this
dashboard checkout or inside the configured `--meco-data-root`. When omitted,
the curriculum endpoint considers `--run-dir` only if it is inside the dashboard
checkout; configuring an external MECO data root does not implicitly redirect
curriculum reads. Other dashboard routes continue to use `--run-dir` as before.

This documents a serving command; it does not claim that a dashboard process is
currently serving this checkout or that its endpoint has been runtime-verified.
