# Bounded large-evaluation dashboard report sidecars

## Planner scope

The scoped dashboard remediation adds bounded compact summaries for large evaluation reports, validates summary/status agreement, and safely publishes/copies sidecars for dashboard projection. Planner source: `docs/plans/dashboard-current-error-remediation-20261025.md`, SHA-256 `adde46e4d4e5952b947ca6521bc118ddf693d5a6f191a752917913210bfb046f`.

## Implementer lineage

- `36b0832b0d72c3b724c7ab51b54751b9afa95e76`: bind the summary to terminal evaluation status.
- `4714bab0b7bbae89c780893ef428c57d0fe87b7e`: bounded summary handling.
- `68fa7b945ae55e1ffc743c4ffb50908ffe271e15`: bound concurrent writers.
- `e0895dbc29cb3c661302c5eb33e14090ac927cee`: reject incomplete and negative metrics.
- `0fbefa10bff2efa1e40196d20759bba0a090813e`: accept partial legacy bout summaries with consistent bounds.
- Integration commit `c55141876f2d6ff2ab1102294da79c222a742219`.
- Scoped paths and SHA-256 values at ledger base `c55141876f2d6ff2ab1102294da79c222a742219`:
  - `frc_defense/dashboard_api.py`: `048ff48c82521f59a9936976ba89ab2f06f9fbf01ae137dad15b3c8fda875be2`
  - `frc_defense/meco_dashboard.py`: `7eee6a15015b5517224eefd2b3e61597371d942e7eec88e8ea09687265ac82b6`
  - `frc_defense/meco_supervisor.py`: `d6e6bfe5a54493683b0103cf7035a1846cff50683af841d1323988cb36dcde24`
  - `frc_defense/meco_track_training.py`: `2c94a4c717d28737f3ff89fd129fc6a5528fd8bc83de2b61a2c3a150ab6debdc`
  - `tests/test_dashboard_curriculum.py`: `c4831a3faf28ee5afb390e35751cbf6de5653667dc537f33db5b5c265916fc19`
  - `tests/test_meco_dashboard.py`: `898fe6a2a9463e93fa7cc4f0bf6651d3dae8ae21ae89419b7fd56da145407cc6`

## Independent review

The exact final feature head `0fbefa10bff2efa1e40196d20759bba0a090813e` was approved after corrections to summary validation. Review covered bounded and sanitized sidecar handling, raw-report and terminal-status SHA binding, exact schema and metric checks, count relationships including partially covered unsuccessful outcomes, safe paths, atomic publication/copies, concurrent writers, and API projection behavior. The added mixed v3/v4 fixture addresses writer-reader compatibility. The implementer reported 58 focused tests passing and a clean diff check; this was not represented as independent reviewer test execution.

## Limits

The sidecar is a bounded dashboard projection, not the full raw report. It supports compact diagnostic summaries and legacy-field projection only; it establishes no model performance or qualification claim. No service currentness is claimed until the post-integration restart and endpoint/asset checks are recorded.
