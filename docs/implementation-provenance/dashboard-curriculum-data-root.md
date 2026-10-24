# Dashboard curriculum external data root — scoped summary

## Planner scope

The planner traced the curriculum status handler and found that it accepted
`run_dir`/`curriculum_root` while omitting the configured MECO data root from
root resolution. As a result, an explicit valid curriculum path under the
configured external data root (the main checkout's live campaign data) was
rejected because the helper confined it to the serving checkout. The planned
contract was to accept explicit paths only beneath the configured data root,
keep implicit paths confined to the checkout, and reject path escapes and
symlinks.

## Implementation

Candidate commit `9597d16e33d18d6dea5ea4f2ee6d6e0715191428` passes the
configured MECO data root through dashboard routing and API resolution. It
allows an explicit curriculum root inside that configured root while preserving
checkout confinement for implicit lookup and rejecting unsafe symlink/path
cases. Changed paths are `docs/MECO_DASHBOARD_DATA_ROOT.md`,
`frc_defense/dashboard.py`, `frc_defense/dashboard_api.py`, and
`tests/test_dashboard_curriculum.py`.

## Independent review, checks, and dashboard observation

The scoped independent reviewer `/root/phase4_start_profile_provenance_review`
approved the exact implementation and its follow-up review. Sixteen focused
tests passed, including HTTP/API acceptance for an external in-root path and
rejection of outside-root, implicit external, and symlink cases; Python
compilation and `git diff --check` passed. After integration, the dashboard
service was restarted and observed active at PID 1928888 on the candidate
checkout. `/api/meco-status`, `/api/curriculum-status`, `/assets/meco.js`, and
`/assets/meco.css` returned HTTP 200, with backend and asset hashes matching
the candidate. The configured data root was the live main checkout. The
curriculum endpoint's generic curriculum summary showed zero runs; five-track
MECO status is reported separately by `/api/meco-status`.

## Limits

This verifies path resolution and service currentness for dashboard endpoints;
it is not a simulator evaluation or replay. No simulator, GPU, training, or
replay was run. No policy qualification or five-track completion is claimed.
