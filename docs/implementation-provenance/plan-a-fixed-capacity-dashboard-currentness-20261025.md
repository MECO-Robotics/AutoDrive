# Plan A fixed-capacity dashboard currentness evidence

Status: **verified** after integrating the reviewed Plan A code, corrected provenance ledger, and digest pin into `main`.

## Checkout and service

- Integrated checkout: `f56f7804bd1616ac3ecf5ca957f9e48b64202bad`.
- The primary worktree retains the pre-existing user edit in `frc_defense/csrc/tensor_collision_multi_hip_kernel.hip`; its SHA-256 remains `e6d02fffa51dba1a21cb6f3d2844c480e567087e92a73324208ce19088a06d6e`. It is outside the integration diff.
- Verification time: 2026-10-25 17:36 UTC.
- Service: user unit `frc-defense-dashboard.service`, active after restart; PID `3779821`, invocation ID `47fd08a8124b498eb5f17be7f97e44b0`, start time `2026-10-25 17:36:19 UTC`.
- Working directory: `/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive`; port `8765`; campaign data root remains the live r4 diagnostic campaign.

## API and provenance checks

After the restart, both endpoints returned HTTP 200:

- `/api/curriculum-status`: response SHA-256 `0f5b8997150209a587768e4bf529c81944159840059d314f84fb53fc8ac39d46`. The provenance status is `scoped-transcript-summaries`, validation status is `validated_summary_only`, and `untrusted_reason` is absent. The response contains 90 tranches and exactly one `plan-a-fixed-capacity-integration` entry.
- `/api/meco-status`: response SHA-256 `1832df3f54ade1ac713ac7b0c65eff08b1f99aa8a46ecfead0a803e7c1c98f62`. It reports five errors in older evaluation records: four oversized cycle reports lack evaluator-summary bindings, and `evaluation-history.json` is incomplete or invalid. These pre-existing report errors do not affect the curriculum ledger validation.

The local ledger SHA-256 is `ed4f2ef24408b352db0d6b9fa51d399e2313bc10a4c2664171443453b5bb3e74`, equal to `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in `frc_defense/dashboard_api.py`. `_EXPECTED_PLAN_SHA256` remains `718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10`.

## Frontend assets

All eight routes returned HTTP 200 and matched their checkout files byte-for-byte:

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

This validates the integrated ledger and dashboard assets only. No replay was generated or reviewed; no simulator behavior, GPU/HIP parity or profiling, training outcome, qualification, or first-stage curriculum passage is established by this check.
