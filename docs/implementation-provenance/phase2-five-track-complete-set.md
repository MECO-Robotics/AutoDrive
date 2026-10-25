# Phase 2 five-track complete-set coordinator

## Scope

Implement the planner-approved offline coordinator for the shared Plan C dataset,
C1/C2 fitted artifacts, Plan A U+B+P bundle, five-track seal, and manifest
warm-start mapping. It validates and resumes completed stages, records immutable
stage receipts, rejects unsafe or untrusted output namespaces, and forwards the
manifest's exact five warm-start paths to manifest-mode training.

## Three-pass evidence

- Planner scope: `docs/plans/phase2-five-track-orchestration-scope.md`, SHA-256
  `ed443dc182f2f88cb049096c173596a7476f3fec7266abd9729a6b1477113859`.
- Implementer branch head reviewed: `d77cb19c0bcd91688f2aa3d0ac4d69c10efcd982`.
  The implementation was built through commits `02f2e3452`, `2b524e3c7`,
  `676a50f11`, `9df2f4692`, `63b41f9bc`, and `41efbb1ca`; test-contract
  follow-up is `d77cb19c0`.
- Independent reviewer approved implementation head `41efbb1ca6cd08c9cf45ad8811088b129130bfbf`
  and test-only head `d77cb19c0bcd91688f2aa3d0ac4d69c10efcd982`.
- Integrated commits on `main`: `b54f903e2`, `f613726ba`, `63121a2f8`,
  `626943310`, `8930429fe`, `15c12265c`, and `59524d21a`.

## Validation

- `PYTHONPATH=. pytest -q tests/test_meco_training_manifest.py tests/test_meco_phase2_complete_set.py` — 43 passed.
- `PYTHONPATH=. pytest -q tests/test_meco_supervisor_dispatch.py -k 'manifest_mode_propagates_phase4_train_exposure'` — 2 passed, 19 deselected. This selection uses a fake supervisor and does not acquire device leases.
- Python compilation and `git diff --check` passed on the implementation worktree.
- Exact-head static review approved the namespace, resume, receipt, manifest
  forwarding, and focused test changes.
- After the ledger and dashboard digest pin were updated, the dashboard was
  restarted on this checkout. `/`, `/api/meco-status`,
  `/api/curriculum-status`, `/assets/meco.js`, and `/assets/dashboard.js`
  returned HTTP 200; the curriculum API reported
  `validated_summary_only` with 67 tranches. Both served JavaScript asset hashes
  matched the checkout. No replay was generated because this tranche did not
  run a simulator behavior evaluation.

## Limits and next gate

The coordinator has not produced a real dataset, warm-start weights, or seal.
No simulator behavior evaluation, fresh replay, GPU collection/fitting,
qualification, or deployment validation was performed. The active r4 campaign
remains on its older diagnostic source and holds both WX9100 leases; it was not
changed. Execute the real Phase 2 campaign only after those leases are released,
then inspect the generated artifacts and run the seeded simulator acceptance
scenario with a current dashboard and fresh replay. Readiness and qualification
remain false until those separate gates pass.
