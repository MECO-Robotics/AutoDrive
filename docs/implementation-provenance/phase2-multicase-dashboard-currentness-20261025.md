# Phase 2 multi-case ledger dashboard currentness evidence

Collected after integrating the ledger digest pin at
`b31d7093e30f001fe1b41f74380adc412921f6c3` on `main`.

## Service and API

- User service `frc-defense-dashboard.service` restarted at
  `2026-10-25T09:03:22Z`; active at verification.
- Invocation ID: `7d15c7a8d2c14de7ab74ccb8f9d95185`; PID `3251559`.
- Working directory: `/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive`.
- Checkout/source commit: `b31d7093e30f001fe1b41f74380adc412921f6c3`.
- `/api/meco-status`: HTTP 200 at `2026-10-25T09:03:28Z`.
- `/api/curriculum-status`: HTTP 200 at `2026-10-25T09:03:28Z`; three-pass
  validation is `validated_summary_only` with 78 tranches. The latest entry is
  `phase2-multicase-five-track-warmstart-coordinator`.
- Ledger SHA-256 and API pin both equal
  `670b4420288cc15d94e7b685a4467a84a868669820dafc080da17dc95adaa472`.

## Frontend asset identity

Each route returned HTTP 200 and its served body matched the local checkout
byte-for-byte; local and served SHA-256 values are identical.

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

The live r4 training service remained active through dashboard restart; its
supervisor PID remained `2750253`. This record verifies ledger serving and
frontend currentness only. It does not claim generated training data,
warm-start readiness, simulator behavior, replay, qualification, or promotion.
