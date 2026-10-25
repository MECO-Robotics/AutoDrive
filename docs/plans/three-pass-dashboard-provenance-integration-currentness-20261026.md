# Pass 1 plan: integrate and verify dashboard provenance summaries

**Pass:** 1 — integration/currentness plan
**Base:** `origin/main` `d30cdd89b4d404d545254423e9d5bc950804e346`
**Approved candidate:** `origin/codex/three-pass-dashboard-provenance-impl-20261026` at `8d6ddca4ce5c12ad2e4d80a79557d8ad7fb54e14`
**Status:** planning only; no merge, commit, push, tests, service operations, devices, runtime, or training actions.

## Objective and boundary

Integrate only the approved dashboard provenance update that registers two scoped summaries and pins the ledger digest. Do not integrate either underlying PPO/reset/pressure source candidate `54bd1dc4651b039e2924e602c2ddf13a165536af` or score-event-buffer source candidate `82009ecdd7b827024928f2ded0ce56ba9889c863`. Their code remains separate, pushed candidates and is not part of this dashboard integration.

The candidate is two commits ahead of the exact base: `283a10b33` persists the summaries/path manifests and `8d6ddca4` registers them and updates the API digest pin. `git diff --name-status origin/main...8d6ddca4ce5c12ad2e4d80a79557d8ad7fb54e14` has the intended scope: `docs/MECO_THREE_PASS_PROVENANCE.json`, `frc_defense/dashboard_api.py`, and four provenance artifacts (two summaries and two path manifests). The integration must preserve these exact artifacts and should not absorb unrelated work.

## Current state and service facts

- Main checkout is `d30cdd89b4d404d545254423e9d5bc950804e346` and has a user-owned modification at `frc_defense/csrc/tensor_collision_multi_hip_kernel.hip`. Keep the main checkout untouched; perform integration in a clean isolated worktree.
- Candidate head is not an ancestor of main. Its merge base is exactly `d30cdd89b4d404d545254423e9d5bc950804e346`.
- The active dashboard serving the campaign is PID `3915965`, running from the AutoDrive checkout with campaign data root `/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDriveCampaigns/meco-five-continuous-20261025-r4`, on port `8765`. At planning time `/api/curriculum-status` and `/api/meco-status` both return HTTP 200. The curriculum payload currently reports `scoped-transcript-summaries` with 92 tranches and lacks the two new requirement IDs.
- The verified owning unit for PID `3915965` and port `8765` is `frc-defense-dashboard.service`; its active override sets `WorkingDirectory` to this AutoDrive checkout and points run/data roots at the r4 campaign. A separate enabled user unit, `frc-defense-deterministic-offense-dashboard.service`, runs PID `2010093` on port `8766` with a different run directory. Do not restart that separate unit. Before any later restart, reverify the active unit, PID, checkout cwd, campaign data roots, listener, and process command line.

The dashboard update request implies verifying currentness after integration. Restart only the exact active r4 dashboard process through its verified owning service/supervisor after merge/push. If its owner cannot be identified safely, stop before restart and report the blocker; do not kill a PID or restart the unrelated port-8766 unit.

## Pass 2 integration procedure

1. Refresh origin refs and verify current `origin/main` is still `d30cdd89b4d404d545254423e9d5bc950804e346`. If it advanced, create a fresh clean HDD-backed integration worktree at the new exact main and rerun all diff, path, digest and service checks; do not force or rewrite history.
2. Create an integration branch/worktree from that exact main. Verify clean status. Preserve the user's dirty HIP file on their existing main checkout; do not stash, reset, copy, or include it.
3. Verify candidate HEAD and its four provenance artifact hashes. Review the complete candidate diff and confirm it changes only the ledger, digest pin, and the two summary/two path-manifest artifacts described above. Confirm the rows point to the exact source candidate SHAs, retain the stated checkout-present path subsets, and do not claim integration, test execution, runtime, or qualification for those source candidates.
4. Perform a normal non-fast-forward merge of `codex/three-pass-dashboard-provenance-impl-20261026` into the clean integration branch. Do not cherry-pick source implementation branches. Inspect the resulting merge diff and conflict resolutions; any conflict touching unrelated files is a stop condition.
5. Recompute the ledger SHA-256 from the merged bytes and compare it exactly with `frc_defense/dashboard_api.py::_EXPECTED_THREE_PASS_PROVENANCE_SHA256`. Confirm plan digest still matches the pinned evaluation-plan bytes. Check all summary/path-manifest paths resolve inside the repo, are regular files, and their SHA-256 values match the row references. Confirm both new IDs are present exactly once; JSON parses; candidate path manifests match their exact full-diff rows and blob IDs; row status distinguishes branch-pushed source candidates from integrated dashboard-ledger code.
6. Commit the normal merge on the isolated integration branch. Push this integration branch as a candidate branch only; do not update `origin/main` yet. Record exact pre-merge main, merge commit SHA, merge parents, candidate branch name, and pushed SHA. Do not include or alter either PPO/reset/pressure or score-buffer source candidate.

## Pass 3 integration review

An independent reviewer checks the exact pushed merge commit, ancestry, full diff, four artifact hashes, both candidate SHAs and row fields, summary wording, plan/ledger/API digest consistency, path-manifest integrity, and that no source candidate was merged. Review is limited to provenance correctness; it must not imply review or approval of the underlying implementation beyond their existing exact-head Pass 3 summaries. Pass 3 must approve the exact merge SHA before `origin/main` is updated.

After exact-merge approval, fetch `origin/main` and require it still equals the reviewed merge base. Only then fast-forward `main` to the reviewed merge SHA and push that fast-forward update from the isolated integration worktree, without checking out or mutating the user’s dirty main worktree. If `origin/main` advanced or the merge candidate changed, stop; rebuild the integration candidate from the new exact main and repeat Pass 2 and Pass 3. Never push an unreviewed merge to main or rewrite remote history.

## Post-merge dashboard currentness

1. Reconfirm exact serving process identity/owner and that it serves the merged checkout. Only restart the active r4 dashboard service/process through that verified owner. Do not touch training service, GPU/device leases, simulator, or the separate port-8766 dashboard.
2. Verify service active and endpoints respond: `/api/curriculum-status` must return HTTP 200 with `three_pass_provenance.status == "validated_summary_only"`, exactly one row each for the PPO/reset/pressure and score-buffer IDs, the expected candidate heads, and no untrusted reason. `/api/meco-status` is a separate campaign endpoint and must independently return HTTP 200.
3. Confirm the dashboard serves frontend assets from the merged checkout and that relevant asset bytes match the checkout. This tranche changes no frontend files; still record the checked asset routes/hashes and backend checkout identity.
4. Record restart time, process/service identity, checkout SHA, endpoint statuses, provenance status and row IDs, plus asset comparison result. Claim dashboard currentness only after these checks pass.

No test suite, training process, simulator, GPU/HIP job, replay, or qualification run is authorized or required for this dashboard-only integration/currentness task. Do not claim candidate runtime or qualification evidence from API validation.

## Acceptance and stop conditions

- The merge is a normal merge commit on a clean integration worktree based on the verified current main; the dirty HIP file remains only in the original main worktree and absent from the merge diff.
- The complete merge diff is limited to the provenance ledger, dashboard digest pin, and the four reviewed provenance artifacts; neither source implementation branch is merged.
- JSON, path containment, exact artifact hashes, candidate path manifests, plan digest, ledger digest, and API pin all agree. The two new rows are valid summary-only evidence, with limits preserved.
- The active campaign dashboard—not the separate port-8766 unit—is restarted only through its verified owner after integration. Both API endpoint checks and frontend checkout/hash check pass before currentness is claimed.
- Stop if main advanced without revalidation, source candidate files enter the merge, any row/hash/path manifest is inconsistent, service ownership or checkout identity is unclear, endpoint status is untrusted, or frontend/backend currentness cannot be verified. Preserve current service and report the specific blocker; do not improvise process control.
