# Diagnostic monitor dashboard projection

## Planner pass

The supervisor publishes a manifest-bound diagnostic monitor summary, but the dashboard queue projection discarded it. The scoped plan adds an allowlisted, freshness-aware diagnostic-only projection and a dashboard card while preserving the finite evaluation schedule display. Stale or malformed queue data must not claim live progress; monitor output remains ineligible for qualification.

Planner commit: `2e9206c518a806c3588b0099503eb290d4d7a653`.
Plan: `docs/plans/diagnostic-monitor-dashboard-scope.md`.

## Implementer pass

Implemented the queue projection and monitor card in `frc_defense/meco_dashboard.py`, `frc_defense/dashboard_static/meco.html`, and `frc_defense/dashboard_static/meco.js`. Added fixture coverage for fresh, absent, malformed, stale, and error monitor states in `tests/test_meco_dashboard.py`.

Implementation commit: `817a6f9076d4c28fea34ba26634458b8f69c5547`.

## Reviewer pass

Independent review approved exact head `817a6f9076d4c28fea34ba26634458b8f69c5547` (parent `2e9206c518a806c3588b0099503eb290d4d7a653`). The reviewer confirmed allowlisted projection, stale/malformed fail-closed behavior, bounded error/history data, no monitor leakage through generic evaluation fields, diagnostic-only qualification semantics, safe `textContent` rendering, and preservation of finite evaluation schedule UI.

## Verification and limits

`PYTHONPATH=. pytest -q tests/test_meco_dashboard.py` passed **32 tests**. Python compilation, `node --check`, and `git diff --check` passed. This dashboard-only tranche did not run a simulator, generate a replay, use a GPU, or alter active training. It exposes diagnostic status; it does not create evaluations or qualification evidence.
