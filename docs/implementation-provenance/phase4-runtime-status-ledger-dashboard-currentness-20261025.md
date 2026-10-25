# Phase 4 runtime-status dashboard currentness evidence

Evidence was collected at `2026-10-25T14:12:55.652320Z` after integrating the
reviewed provenance update.

## Review, integration, and source identity

- First Pass 3 approved the pending candidate
  `cfadae21999fdbd31597fd096bc916485d20ec98`.
- Second Pass 3 approved finalizer
  `637cc4b64fd5d1b22532305dd86fc14deb66a913` against plan
  `473a829c9be3923554b8b87c252d6ea9624ef7e2`.
- The finalizer and fast-forward integration commit on `main` and
  `origin/main` are both `637cc4b64fd5d1b22532305dd86fc14deb66a913`.
- The running dashboard checkout is this commit. The dashboard source files
  are unchanged from it: `frc_defense/dashboard.py` SHA-256
  `220e2cff382ac4c2faff02d81de37b1d0e7fd25b3e6c9988d9b15a2feb5b1a33`
  and `frc_defense/dashboard_api.py` SHA-256
  `8f823b2aa8bb8fbd3a53d012d741ff62e9e28b6b3e760551370fe6c1aa0cfd64`.

## Service and API

`frc-defense-dashboard.service` was restarted at `2026-10-25 14:09:48 UTC`
and was active at evidence collection. PID: `3573611`. Invocation ID:
`40e11aad77c14fd596cd6452a508087c`. Working directory:
`/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive`.

Both `/api/meco-status` and `/api/curriculum-status` returned HTTP 200. The
curriculum API's `three_pass_provenance` reports
`validation_status=validated_summary_only`, stored `status=scoped-transcript-summaries`,
88 tranches, latest requirement ID
`phase4-runtime-status-ledger-dashboard`, and no `untrusted_reason` field.
The local ledger SHA-256
`a7c26f2c9c6d54bf79ac127f39d1934b52b361e22e4e5470864144ecaaaa57a3` matches
`_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in `frc_defense/dashboard_api.py`.

The successful MECO status response also reports four separate data-read
errors: cycle files `cycle-000001-263b6684.json`,
`cycle-000002-6e9cca09.json`, and `cycle-000003-b7546895.json` lack evaluator
summary bindings; `evaluation-history.json` is incomplete or invalid. These
do not change the reported three-pass provenance validation state.

## Frontend route and asset identity

All eight required routes returned HTTP 200. Every served response SHA-256
matched its corresponding file in the same checkout byte-for-byte.

| Route | Checkout file | SHA-256 |
| --- | --- | --- |
| `/` | `frc_defense/dashboard_static/index.html` | `77e7c1fe95db0cac671316eda480dbadffafe6bdeb5c706b9d3b9d0b9b0a5ae0` |
| `/meco` | `frc_defense/dashboard_static/meco.html` | `59d43b1e2c8875d6937530874b021221a6c87900cd978447f628719a8f3d7fe2` |
| `/assets/dashboard.css` | `frc_defense/dashboard_static/dashboard.css` | `ce3cde2f21c9243abeaa67b86ef69b015c72612c6e2c3428f044345244084c72` |
| `/assets/dashboard.js` | `frc_defense/dashboard_static/dashboard.js` | `487dff5cfb48d4d7e6d419be343d5e1149c291b4ba9866d227fa312972867219` |
| `/assets/meco.css` | `frc_defense/dashboard_static/meco.css` | `5b71d2c3f9ee0eb9f5828bb448b6b211d811a04f13c1bca32626c92d690a8a05` |
| `/assets/meco.js` | `frc_defense/dashboard_static/meco.js` | `51a67f3f77d38b23dad046b2189921724417ccc9cd581d3f9d8864573b32f3a2` |
| `/assets/field_rendering.js` | `frc_defense/dashboard_static/field_rendering.js` | `d9677094b2d55efb5cdab6416b99d1e85c9acd1086314bcab745f3b57d83646c` |
| `/assets/training_rendering.js` | `frc_defense/dashboard_static/training_rendering.js` | `500b29c6a0fb679762d668a7bd2224020ec18c16f41bdf7577922d20eaf03950` |

The active r4 diagnostic training campaign was left running and unmodified.
This verifies provenance and dashboard serving only. No simulator run or replay
was generated or reviewed, and no Phase 4 behavior outcome or qualification
claim follows from this check.
