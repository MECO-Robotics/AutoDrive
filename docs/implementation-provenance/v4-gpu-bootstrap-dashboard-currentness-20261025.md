# V4 GPU bootstrap dashboard currentness check

Checked at **2026-10-25 18:55 UTC** after integrating the reviewed ledger and digest pin.

- Checkout: `9dce3c46d50cbbee0e043e4b7061c3e66cd78b23` (`main`)
- Service: `frc-defense-dashboard.service`, active
- Invocation: `0987d00cc6fe40dbb0577629ff2dd4a5`
- PID: `3867236`
- Working directory: `/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive`
- Dashboard: `http://127.0.0.1:8765`
- Ledger SHA-256: `0386090213c81e0aa0bcb1371f639d39eb7d36a3714bde6b6ecb6122c5561758`
- Dashboard API source pin `_EXPECTED_THREE_PASS_PROVENANCE_SHA256`: same digest above. `_EXPECTED_PLAN_SHA256` was not changed by this tranche.

## Backend

Both endpoints returned HTTP 200 after service restart:

- `/api/curriculum-status`: `three_pass_provenance.validation_status` was `validated_summary_only`; 91 tranches were present, including `v4-gpu-cluster-bootstrap` exactly once, with no `untrusted_reason`. The API does not return the manifest SHA as a response field; the served checkout's ledger file hash and source pin were verified against the digest above. The curriculum endpoint returned no data errors.
- `/api/meco-status`: HTTP 200, with no top-level error. It reported five unrelated evaluation-data errors: four cycle reports lacked evaluator summary binding and `evaluation-history.json` was incomplete or invalid. These are reported as data-read issues, not ledger validation failures.

## Frontend assets

Every route returned HTTP 200. Served bodies matched the corresponding files in this checkout byte-for-byte.

| Route | Local file | Served SHA-256 | Match |
| --- | --- | --- | --- |
| `/` | `frc_defense/dashboard_static/index.html` | `77e7c1fe95db0cac671316eda480dbadffafe6bdeb5c706b9d3b9d0b9b0a5ae0` | yes |
| `/meco` | `frc_defense/dashboard_static/meco.html` | `59d43b1e2c8875d6937530874b021221a6c87900cd978447f628719a8f3d7fe2` | yes |
| `/assets/dashboard.css` | `frc_defense/dashboard_static/dashboard.css` | `ce3cde2f21c9243abeaa67b86ef69b015c72612c6e2c3428f044345244084c72` | yes |
| `/assets/dashboard.js` | `frc_defense/dashboard_static/dashboard.js` | `487dff5cfb48d4d7e6d419be343d5e1149c291b4ba9866d227fa312972867219` | yes |
| `/assets/meco.css` | `frc_defense/dashboard_static/meco.css` | `5b71d2c3f9ee0eb9f5828bb448b6b211d811a04f13c1bca32626c92d690a8a05` | yes |
| `/assets/meco.js` | `frc_defense/dashboard_static/meco.js` | `11ac3ec71ebfd43e0d1dab9bb47b3e68455c90830a98acfaf86519769d925112` | yes |
| `/assets/field_rendering.js` | `frc_defense/dashboard_static/field_rendering.js` | `d9677094b2d55efb5cdab6416b99d1e85c9acd1086314bcab745f3b57d83646c` | yes |
| `/assets/training_rendering.js` | `frc_defense/dashboard_static/training_rendering.js` | `500b29c6a0fb679762d668a7bd2224020ec18c16f41bdf7577922d20eaf03950` | yes |

No replay, simulator evaluation, tests, GPU execution, training operation, calibration, or qualification was performed for this dashboard check. The training campaign remained under its existing supervisor.
