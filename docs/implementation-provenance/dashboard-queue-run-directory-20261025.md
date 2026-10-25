# Dashboard curriculum queue run-directory handling

## Planner scope

The scoped work adds queue run-directory handling that preserves containment and symlink protections while keeping legacy `directory` compatibility. The approved behavior accepts a legacy absolute `directory` only when it resolves inside the selected root without traversing symlinks. `run_directory` remains relative-only. Outside-root paths, traversal, symlink paths, and conflicting directory fields fail closed.

The original scoped planner/task and reviewer correction/approval messages are the source records for this summary; no standalone queue planner or reviewer artifact was persisted before this ledger update.

## Implementer lineage

- Feature commit `ea6f039ced9c1d249fea7059ec2afc94ef3c27e7`: safely read `run_directory`.
- Corrective and final feature commit `8c9b6bdf80ca7b97a392c88e8b41742607b80182`: restore safe in-root absolute legacy `directory` values.
- Integration commit `ff5c862b183d71c733e0094b186fb8fbf9042904`.
- Scoped paths and SHA-256 values at ledger base `c55141876f2d6ff2ab1102294da79c222a742219`:
  - `scripts/report_curriculum_progress.py`: `7a5aacc4f89478479fb186621b68d9d1e0df9766780862799bbd8802d5cca952`
  - `tests/test_dashboard_curriculum.py`: `c4831a3faf28ee5afb390e35751cbf6de5653667dc537f33db5b5c265916fc19`

## Independent review

The exact final feature head `8c9b6bdf80ca7b97a392c88e8b41742607b80182` was approved. The earlier candidate was returned for changes because it rejected safe absolute in-root legacy `directory` paths. The corrected review confirmed that compatibility is restored while traversal, symlink, outside-root, and conflict cases remain fail-closed; test cases distinguish safe in-root absolute paths from unsafe outside paths. No dashboard service currentness or queue liveness result is claimed here.

## Limits

This is scoped source and review provenance. It does not establish that training is live, that any campaign is running, or that the dashboard service is current. Post-update service and asset checks remain pending.
