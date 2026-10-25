# Plan C dashboard currentness evidence

Status: **verified** after integrating the approved finalizer and restarting the
dashboard from the integrated checkout.

## Integration and service

- Code review candidate C: `cdeb9391bf23e5c2460527959132008aee6f8eb1`,
  approved at code/test-source review level.
- Ledger candidate D: `8d64233d4573600321956c74391842e7623f18f7`,
  independently approved for ledger correctness.
- Finalizer E: `0ce905a77cd5d7ae92dea6917999cf034892a6ea`, approved by the
  second exact-head Pass 3 review.
- Integrated code and ledger checkout: `8e0ad89aad4721d9f58bcfb50e27d8394f2f0fa0`
  (tree `7c58a4f9ef336a38cfc7f5cc6fc33774cf72c9a2`).
- Verification time: 2026-10-25 16:04 UTC.
- Service: `frc-defense-dashboard.service`, active and running; PID `3689208`,
  invocation ID `49d51d88826b4251817f1ea6149fea51`, start time 2026-10-25
  16:04:02 UTC.
- Working directory: `/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive`.
- Dashboard port: `8765`; campaign data root remains the active r4 diagnostic
  campaign directory.

## API and ledger checks

Both API routes returned HTTP 200 after the restart:

- `/api/curriculum-status`: body SHA-256
  `f88d0e02ac2c4a92956439172cf35962d7cf9a10bb44165feada5d654dec9180`.
  `three_pass_provenance.validation_status` is `validated_summary_only`;
  `three_pass_provenance.status` is `scoped-transcript-summaries`; the response
  contains 89 rows and exactly one `meco-plan-c-route-device-residency` row;
  `untrusted_reason` is absent. Its `errors` list is empty.
- `/api/meco-status`: body SHA-256
  `9233bb3ccf1757d8490f9f57a0b6aaff65e50e45f726bc7d0309cd05264aa08b`.
  The route returned successfully. It also reported five unrelated errors in
  older campaign reports: four oversized cycle reports lack evaluator-summary
  bindings, and `evaluation-history.json` is incomplete or invalid. These data
  read errors do not affect the curriculum provenance validation.

The integrated manifest SHA-256 is
`fa12a6159008a3b52febb651f3ce4a4715250261190f2320e2e6eb437f7a400a`, matching
`_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in the same checkout.

## Frontend assets

Each route returned HTTP 200 and its response body matched the corresponding
file byte-for-byte in the integrated checkout.

| Route | Checkout file | Served SHA-256 |
|---|---|---|
| `/` | `frc_defense/dashboard_static/index.html` | `77e7c1fe95db0cac671316eda480dbadffafe6bdeb5c706b9d3b9d0b9b0a5ae0` |
| `/meco` | `frc_defense/dashboard_static/meco.html` | `59d43b1e2c8875d6937530874b021221a6c87900cd978447f628719a8f3d7fe2` |
| `/assets/dashboard.css` | `frc_defense/dashboard_static/dashboard.css` | `ce3cde2f21c9243abeaa67b86ef69b015c72612c6e2c3428f044345244084c72` |
| `/assets/dashboard.js` | `frc_defense/dashboard_static/dashboard.js` | `487dff5cfb48d4d7e6d419be343d5e1149c291b4ba9866d227fa312972867219` |
| `/assets/meco.css` | `frc_defense/dashboard_static/meco.css` | `5b71d2c3f9ee0eb9f5828bb448b6b211d811a04f13c1bca32626c92d690a8a05` |
| `/assets/meco.js` | `frc_defense/dashboard_static/meco.js` | `51a67f3f77d38b23dad046b2189921724417ccc9cd581d3f9d8864573b32f3a2` |
| `/assets/field_rendering.js` | `frc_defense/dashboard_static/field_rendering.js` | `d9677094b2d55efb5cdab6416b99d1e85c9acd1086314bcab745f3b57d83646c` |
| `/assets/training_rendering.js` | `frc_defense/dashboard_static/training_rendering.js` | `500b29c6a0fb679762d668a7bd2224020ec18c16f41bdf7577922d20eaf03950` |

This verifies the current three-pass ledger and frontend assets from the
integrated checkout. No replay was generated or reviewed. It establishes no
simulator outcome, GPU/HIP parity or profiling, training result, qualification,
or whole-system GPU-only execution.
