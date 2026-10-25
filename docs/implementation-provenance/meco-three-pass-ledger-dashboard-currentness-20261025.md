# Phase 4 ledger dashboard currentness evidence

Collected 2026-10-25 after integrating ledger commit
`65bc56e986facc19949852b6c942ccc8bd09179f` on `main`.

## Service and API

- User service: `frc-defense-dashboard.service`, active after restart at
  `2026-10-25T08:50:28Z`.
- Invocation ID: `4cd8eee391944e2bac7dd5d8c9a4c7d0`; PID `3238088`.
- Working directory: `/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive`.
- `/api/meco-status`: HTTP 200 at `2026-10-25T08:51:49Z`.
- `/api/curriculum-status`: HTTP 200 at `2026-10-25T08:51:50Z`; three-pass
  validation is `validated_summary_only`, with 77 tranches. The final entry is
  `phase4-reactive-pool-outcome-bridge`.
- Ledger SHA-256 and dashboard API pin:
  `1998ee1cffb6c657c0667dc55c1ec7d9d9fb51f9f5ade3397a5e88050387c532`.
- The MECO status response also contains two existing oversized-cycle-file
  read errors (`cycle-000001-263b6684.json` and
  `cycle-000002-6e9cca09.json`); both endpoints returned successfully. These
  data-read errors are separate from the three-pass ledger validation.

## Frontend asset identity

Each served body returned HTTP 200 and matched the listed local checkout file
byte-for-byte (local and served SHA-256 values are identical).

| Route | Checkout file | SHA-256 |
|---|---|---|
| `/` | `frc_defense/dashboard_static/index.html` | `77e7c1fe95db0cac671316eda480dbadffafe6bdeb5c706b9d3b9d0b9b0a5ae0` |
| `/meco` | `frc_defense/dashboard_static/meco.html` | `59d43b1e2c8875d6937530874b021221a6c87900cd978447f628719a8f3d7fe2` |
| `/assets/dashboard.css` | `frc_defense/dashboard_static/dashboard.css` | `ce3cde2f21c9243abeaa67b86ef69b015c72612c6e2c3428f044345244084c72` |
| `/assets/dashboard.js` | `frc_defense/dashboard_static/dashboard.js` | `487dff5cfb48d4d7e6d419be343d5e1149c291b4ba9866d227fa312972867219` |
| `/assets/meco.css` | `frc_defense/dashboard_static/meco.css` | `5b71d2c3f9ee0eb9f5828bb448b6b211d811a04f13c1bca32626c92d690a8a05` |
| `/assets/meco.js` | `frc_defense/dashboard_static/meco.js` | `51a67f3f77d38b23dad046b2189921724417ccc9cd581d3f9d8864573b32f3a2` |
| `/assets/field_rendering.js` | `frc_defense/dashboard_static/field_rendering.js` | `d9677094b2d55efb5cdab6416b99d1e85c9acd1086314bcab745f3b57d83646c` |
| `/assets/training_rendering.js` | `frc_defense/dashboard_static/training_rendering.js` | `500b29c6a0fb679762d668a7bd2224020ec18c16f41bdf7577922d20eaf03950` |

This is metadata and serving-currentness evidence only. It does not claim a
simulator run, replay, training result, qualification, or promotion.
