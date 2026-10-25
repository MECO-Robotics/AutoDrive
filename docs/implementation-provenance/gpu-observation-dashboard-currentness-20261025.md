# GPU observation integration dashboard currentness

Checked after integration at **2026-10-25 19:15 UTC**.

- Serving checkout: `c0a696524` (`main` at restart)
- Service: `frc-defense-dashboard.service`, active
- Invocation: `4fa217ec08ee401f828ed944daae3eca`
- PID: `3891365`
- Working directory: `/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive`
- Base URL: `http://127.0.0.1:8765`
- No replay or simulator evaluation was generated for this source-currentness check.

## Backend

Both endpoints returned HTTP 200.

- `/api/curriculum-status`: `validated_summary_only`, 91 provenance rows, no top-level error, no endpoint errors. Campaign runs were A1 complete (600), A2 complete (600), A3 running (522), C1 running (572), C2 queued (500). All five remain diagnostic and none is curriculum-qualified.
- `/api/meco-status`: HTTP 200 with no top-level error. Six existing data-read errors remain: five cycle reports lack evaluator summary binding and `evaluation-history.json` is incomplete or invalid. These do not affect ledger validation.

## Frontend assets

All routes returned HTTP 200 and the served bodies matched the corresponding files byte-for-byte at the serving checkout.

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

This records dashboard/API/frontend currentness only. It does not establish simulation behavior, fresh replay outcomes, GPU residency, training qualification, or completion of the evaluation plan.
