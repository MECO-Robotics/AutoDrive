# Phase 0 ledger dashboard currentness plan

**Status:** Planned only. This is not evidence that the dashboard was restarted
or that its APIs/assets are current.

After the corrected implementation and the 80-row ledger/pin are integrated on
`main`, the integrator should:

1. Restart the dashboard service from the integrated checkout and verify the
   service is active and its endpoint responds.
2. Fetch `/api/curriculum-status` and `/api/meco-status`. Confirm the
   three-pass provenance is `validated_summary_only`, has 80 rows, and includes
   `phase0-baseline-heldout-contracts` with the integrated commit and exact
   summary artifact hashes.
3. Record any saved-evidence verification errors returned by
   `/api/curriculum-status` as separate current-data errors. The integrator
   reported that this endpoint currently has saved-evidence verification
   errors; that report has not been rechecked in this docs-only plan.
4. Fetch `/`, `/meco`, `/assets/dashboard.css`, `/assets/dashboard.js`,
   `/assets/meco.css`, `/assets/meco.js`, `/assets/field_rendering.js`, and
   `/assets/training_rendering.js`. Compare each served body byte-for-byte with
   the corresponding file in the integrated checkout and record SHA-256
   digests.

Do not treat the earlier 79-row currentness note as post-update evidence. No
replay, simulator run, training run, device/GPU use, or qualification check is
part of this metadata currentness task. If a required service/API/asset check
cannot be completed, report it as unavailable and leave currentness
unverified.
