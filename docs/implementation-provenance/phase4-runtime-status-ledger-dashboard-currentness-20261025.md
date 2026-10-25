# Phase 4 runtime-status dashboard currentness evidence

Verified 2026-10-25 after the reviewed provenance update was integrated at
`637cc4b64fd5d1b22532305dd86fc14deb66a913`.

## Review and integration

- Pass 3 approved finalization commit
  `637cc4b64fd5d1b22532305dd86fc14deb66a913` against plan
  `473a829c9be3923554b8b87c252d6ea9624ef7e2`.
- The finalizer records the first reviewed candidate and does not put its own
  integration SHA or the second review verdict in the digest-pinned ledger.
- The approved integration is present on `main` and `origin/main` at the same
  SHA.

## Dashboard service and endpoint check

After integration, `frc-defense-dashboard.service` was restarted at
`2026-10-25 14:09:48 UTC` and verified active (PID `3573611`) with working
directory `/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive`.
The following routes returned HTTP 200:

- `/`
- `/api/meco-status`
- `/api/curriculum-status`
- `/assets/meco.js`
- `/assets/dashboard.js`

The curriculum API reported `validated_summary_only`, 88 provenance tranches,
and latest requirement ID `phase4-runtime-status-ledger-dashboard`. The local
ledger SHA-256
`a7c26f2c9c6d54bf79ac127f39d1934b52b361e22e4e5470864144ecaaaa57a3` matches
`_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in `frc_defense/dashboard_api.py`.
Served frontend assets matched the integrated checkout byte-for-byte:

- `meco.js`: `51a67f3f77d38b23dad046b2189921724417ccc9cd581d3f9d8864573b32f3a2`
- `dashboard.js`: `487dff5cfb48d4d7e6d419be343d5e1149c291b4ba9866d227fa312972867219`

This verifies the dashboard provenance and assets only. No simulator run or
replay was generated or reviewed, and no Phase 4 behavior outcome or
qualification claim follows from this check. The active r4 diagnostic training
campaign was left running and unmodified.
