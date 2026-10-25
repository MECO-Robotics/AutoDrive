# Pass 1 plan: register approved PPO and score-buffer tranches in dashboard provenance

**Pass:** 1 — planner scope
**Base:** `origin/main` `d30cdd89b4d404d545254423e9d5bc950804e346`
**Scope:** summary-only dashboard provenance for two independently approved, branch-pushed implementation candidates.
**Status:** plan only; no ledger/API edits, tests, services, devices, runtime, or training actions.

## Decision

Add separate scoped summaries for the cumulative PPO/reset/pressure implementation at `54bd1dc4651b039e2924e602c2ddf13a165536af` and GPU score-event-buffer implementation at `82009ecdd7b827024928f2ded0ce56ba9889c863`. Neither candidate is an ancestor of current `origin/main` (`d30cdd89b4d404d545254423e9d5bc950804e346`); therefore report them as pushed feature candidates, not integrated implementation. Do not claim main contains them or that current dashboard serves them.

Do not add the Stage 1 qualification plan as a tranche: it is a plan artifact without a Pass 2 implementation candidate or Pass 3 implementation review.

## Exact candidate evidence

### PPO, selected reset/cadence, pressure state, and checkpoint v5

- Candidate branch: `origin/codex/five-track-pressure-snapshot-v2-pass2-20261026`.
- Exact approved implementation head: `54bd1dc4651b039e2924e602c2ddf13a165536af`.
- The implementation is cumulative, beginning with device-batched PPO (`4340cde85`) and continuing through replay/checkpoint validation, selected-row reset/RNG isolation, row-local cadence, pressure reset identity, and snapshot-v2 static pressure phase preservation. The ledger summary must identify the exact final reviewed head and list implementation files only, not inherited plan documents. In `implementer.paths`, list only the candidate-changed production paths that also exist in the current main checkout; label this list as a “checkout-present path subset” in the row summary and never imply it is the full candidate diff. At this base, `frc_defense/meco_ppo_batch.py` is candidate-only and must not be put in `implementer.paths` because it is absent from main.
- Source paths changed from this plan base are `frc_defense/_tensor_3v3_actions.py`, `_tensor_3v3_gamepieces.py`, `_tensor_3v3_observation.py`, `architecture_scenarios.py`, `curriculum_metrics.py`, `curriculum_scenarios.py`, `meco_baseline_freeze.py`, `meco_phase4_context.py`, `meco_plan_a_policy.py`, `meco_ppo_batch.py`, `meco_track_training.py`, `meco_tracks.py`, `tensor_3v3.py`, `tensor_defense_gamepieces.py`, `tensor_defense_observation.py`, `tensor_defense_opponent.py`, `tensor_defense_reset.py`, `tensor_defense_step.py`, `tensor_physics.py`; `tests/test_meco_phase4_context.py` is a test path, not production behavior evidence.
- Review basis: the parent reported in this conversation that Pass 3 approved exact head `54bd1dc...`. The available report establishes the exact-head verdict only; it does not supply a detailed review checklist. A persisted summary may state that verdict and only the review scope actually reported. Do not infer or invent review checks from commit messages. If project policy requires a source review artifact beyond these reported records, stop rather than claim one.
- Test status: test changes are present in candidate history, but no executed test result is established by the available exact-head evidence in this plan. Record tests as “authored; execution not evidenced” unless a retained exact-head test report is located and hash-bound.

### GPU-resident score-event buffer

- Candidate branch: `origin/codex/gpu-score-event-buffer-impl-20261026`.
- Exact approved implementation head: `82009ecdd7b827024928f2ded0ce56ba9889c863`.
- It depends on and includes the PPO/reset/pressure candidate line; retain its own independent requirement ID and candidate head rather than folding it into the PPO row.
- Its implementation paths relative to the PPO candidate are `frc_defense/gamepiece_actions.py`, `frc_defense/fuel_physics_runtime.py`, `frc_defense/meco_simulated_score_ledger.py`, `frc_defense/meco_track_training.py`, `frc_defense/_tensor_3v3_gamepieces.py`, `frc_defense/tensor_defense_gamepieces.py`, and `tests/test_meco_gpu_score_event_buffer.py`. For `implementer.paths`, include only candidate-changed production paths present in current main, label them as a “checkout-present path subset,” and never imply completeness. Exclude plans and test paths from that field; preserve every changed path in the complete candidate manifest below.
- Pass 1 plan artifact is `docs/plans/gpu-resident-score-event-buffer-20261026.md` at plan branch `origin/codex/gpu-score-event-buffer-plan-20261026`, commit `47ff1037d`; SHA-256 `3095cbee304f21d1d84ae91c1152722f955df47fba58f2399d3e6302d518eff1`.
- Review basis: the parent reported in this conversation that Pass 3 approved exact head `82009ec...`. The available report establishes the exact-head verdict only; it does not supply a detailed review checklist. A persisted summary may state that verdict and only the review scope actually reported. Do not infer or invent review checks from commit messages. If project policy requires a source review artifact beyond these reported records, stop rather than claim one.
- Test status: `tests/test_meco_gpu_score_event_buffer.py` is authored in the candidate. No executed test result is established here; say “authored; execution not evidenced” unless an exact-head report is located and hash-bound.

## Pass 2 implementation boundary

In a separate implementation worktree based on the then-current main, edit only:

1. `docs/MECO_THREE_PASS_PROVENANCE.json` — add exactly two unique summary-only tranche rows and update scope/artifact-origin/limits language to distinguish approved pushed candidates from integrated main. Each row must identify its plan summary, exact implementation candidate commit, implementation paths, persisted reviewer summary, and reviewer-approved exact candidate commit.
2. `frc_defense/dashboard_api.py` — update `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` to the exact SHA-256 of the resulting ledger bytes. No other dashboard behavior or service files are in scope.
3. Add one immutable candidate path-manifest artifact per implementation row under `docs/implementation-provenance/`. Each must contain the complete output of `git diff --name-status origin/main...<full-40-char-candidate-SHA>` and, for every listed path, the candidate blob SHA-1 (or `absent` for a deletion) plus the base blob SHA-1 when applicable. Label each path `checkout-present` or `candidate-only` according to exact path existence in the base checkout. Do not omit documentation/test changes from the complete manifest. Link each manifest from its ledger row (for example, `implementer.candidate_path_manifest_artifact` with repository-relative path and SHA-256), and include its SHA-256 in the ledger. Add two scoped implementation/review summaries only if traceable to the actual Pass 2/Pass 3 records and accurately state test/runtime evidence as absent or unverified. If authoritative review records cannot be recovered/persisted, stop and leave that tranche out; never synthesize approval details.

Do not edit the evaluation plan, Stage 1 plan, implementation source, test source, training checkout/config/checkpoints, dashboard assets, or live service. No cherry-pick/merge or candidate integration is authorized by this plan.

## Required acceptance evidence

- The ledger remains `meco-three-pass-provenance-v2`, `scoped-transcript-summaries`, `evidence_class: scoped_transcript_summary`; plan path and plan digest remain exactly bound to the evaluation plan. Both IDs are unique and role schemas satisfy `_validate_three_pass_manifest`.
- For each row, the planner artifact is a real repo-local file with exact SHA-256 and transcript-summary status; implementation commit is the full 40-character candidate SHA; file paths are safe, unique, present, and limited to implementation changes; reviewer artifact is a real repo-local file with exact SHA-256, review-summary status, verdict `approved`, and `approved_commit` exactly equal to the final candidate head.
- Each row explicitly states `branch-pushed; not integrated into origin/main at base d30cdd89...` (or the newer exact integration base if Pass 2 rebases), and says tests were authored but execution is not evidenced unless authoritative reports prove otherwise. Confirm each row has at least one candidate-changed production path that exists in the exact main checkout; if not, stop that row. Verify `implementer.paths` contains only that checkout-present subset, and verify the linked path manifest contains every changed path with correct status and blob hashes. Dashboard validation of `implementer.paths` establishes only that subset; the immutable manifest documents the complete candidate diff and does not imply candidate source is present or integrated on main.
- Limits explicitly deny claims of successful training, uninterrupted campaign compatibility, source/checkpoint continuity, hardware/HIP execution or parity, simulator/replay outcomes, current dashboard serving candidate code, score-approval completion, promotion, qualification, and whole-plan completion.
- Stage 1 remains excluded because it has no implementation/review row.
- Perform non-test integrity checks only: parse the final ledger as JSON; verify exact uniqueness and required schema fields for the two rows; recompute SHA-256 for the ledger and every referenced planner/reviewer/path-manifest summary artifact; confirm each referenced artifact exists inside the repository and its bytes match the recorded digest; compare each path manifest against `git diff --name-status origin/main...<candidate SHA>` and verify candidate/base blob hashes and checkout-presence labels; confirm `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in `frc_defense/dashboard_api.py` exactly equals the final ledger SHA-256. Do not run tests or services as part of this source-only tranche.
- Only after a separately approved integration/currentness pass may the dashboard be restarted and `/api/curriculum-status` checked for provenance. `/api/meco-status` is the separate campaign endpoint. Such restart/currentness is not claimed by this tranche.

## Risks and stop conditions

The current branch candidates are approved but not integrated. A future main advance can change source-path availability, the plan digest, ledger digest, or reviewer artifact availability; rebase only in the isolated worktree and revalidate all exact hashes. The ledger's repository-path validator rejects missing artifacts and unsafe paths. If review evidence is only a transient message and cannot be persisted with reliable contents, do not make the approval visible as a dashboard-validated row. This tranche records provenance; it does not establish qualification readiness or runtime success.
