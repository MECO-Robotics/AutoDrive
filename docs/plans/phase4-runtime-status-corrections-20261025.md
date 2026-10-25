# Phase 4 runtime-status row corrections

## Pass 1 scope and base

**Planning base:** `origin/main` at
`252eb3d65e27a0d96352cf2abb83265384ce1d8d`.

The focal-pose status correction is already integrated at this base. This plan
covers only the remaining stale or incomplete status rows in
`docs/MECO_PHASE4_RUNTIME_STATUS.md`: stationary-blockage event text, direct
pursuit, trench interception, and legal team starts. All other rows remain
unchanged. The corrections describe implemented code paths and retain the
explicit distinction between code-supported outcomes and accepted simulator
evidence.

## Source audit anchors

The machine-readable row bindings are in
`frc_defense/meco_phase4_runtime_adapter.py`:

- Stationary blockage: `:60-69` declares the assigned-blocker runtime event,
  but says simulator placement/outcome are unvalidated and required outcomes
  are non-authoritative.
- Direct pursuit: `:70-82` declares the schedule and applied-slot events, the
  contact/progress capture, and required outcome sources. It explicitly says
  no held-out simulator replay/dashboard review has been accepted.
- Trench interception: `:83-98` declares schedule/application and contact/
  crossing events, outcome signals, and the narrow candidate-cell gate. The
  14-category domain remains unqualified; the candidate cell needs exact
  11-policy paired coverage and fresh replay/dashboard review.
- Legal team starts: `:126-148` declares reset application, actual-world
  legality and paired outcome sources. Candidate execution requires one
  profile cell, every expected world legal, and both the scenario-defined and
  assigned-profile state IDs. No held-out report/replay review has been
  accepted.

Implementation paths supporting those declarations:

- `frc_defense/curriculum_scenarios.py:170-205,340-366` validates scheduled
  opponent slots and supplies the direct-pursuit policy with sensed target
  state. `:380-389,1010-1027` validates and applies legal-team start
  assignments after scenario setup and before the initial observation.
- `frc_defense/meco_track_training.py:3434-3459,4288-4300` constructs direct
  pursuit and trench episode captures, then emits their records.
- `frc_defense/meco_phase4_direct_pursuit.py:20-56,190-296,350-434` captures
  simulator-confirmed pursuit-pair contact and fixed-objective progress; its
  paired reducer derives time loss only from uncensored same-seed control and
  treatment threshold crossings.
- `frc_defense/meco_phase4_trench_interception.py:336-410,507-648` binds slot
  0 and captures slot-0/3 contact, full-horizon truth-pose samples, pinned
  opening geometry, route crossings, and paired outcomes.
- `frc_defense/meco_phase4_context.py:117-210,375-457` defines protocol-bound
  start-profile assignments, applies profiles to physical pose at reset, and
  checks the poses actually applied for field bounds, chassis overlap, and
  solid-field collisions.
- `frc_defense/meco_validation_campaign_runner.py:221-398` constructs the
  exact start-profile validation pair, validates report coverage and legality,
  and emits a paired-outcomes artifact. It marks behavior validation false
  and qualification ineligible pending replay review.

Relevant existing contract coverage includes
`tests/test_meco_phase4_opponents.py:89-185` for opponent policies and
validation schedules, `tests/test_meco_phase4_design.py:180-206` for registered
Phase 4 case coverage, `tests/test_meco_phase4_campaign_plan.py:178-334` for
reactive/trench schedule dispatch, and `:336-493` for start-profile paired
dispatch and intake. Start-profile pose legality and assignment contracts are
also covered in `tests/test_meco_phase4_context.py:35-178`.
`tests/test_meco_phase4_campaign_plan.py:178-334` uses a fake runner; it is not
simulator evidence. No dedicated direct-pursuit or trench recorder/reducer test
file was found in this audit. No tests were run for this plan.

## Pass 2 exact documentation edits

Update only table rows 16-18 and 21 in
`docs/MECO_PHASE4_RUNTIME_STATUS.md`; keep the already-corrected focal-pose row
and all other rows unchanged.

1. **Stationary blockage, row 16:** keep the route-goal lifecycle endpoint as a
   non-authoritative diagnostic; add
   `phase4_runtime_metrics.stationary_blockage_assigned` as the assignment
   event. State that slot-0 blocker assignment is wired, simulator placement
   remains unvalidated, and the three required registry outcomes remain
   non-authoritative/unavailable for qualification.
2. **Direct pursuit, row 17:** replace `none` with
   `phase4_opponent_schedule`, the exact applied-slot event
   `records[].phase4_direct_pursuit.applied_slots`, and
   `records[].phase4_direct_pursuit.contact_events + objective_route.progress_intervals`.
   Replace “validation slot assignment is not wired; signals unavailable” with
   the bounded source fact: the validation-bound slot assignment, simulator
   event/progress capture, and exact paired outcome reducer are implemented;
   no held-out simulator/replay/dashboard review has been accepted, so
   executable promotion remains fail-closed.
3. **Trench interception, row 18:** replace `none` with
   `phase4_opponent_schedule`, the exact applied-slot event
   `records[].phase4_opponent_policy.runtime.applied_slots`, and
   `records[].phase4_trench_interception.contact_events + crossing_events`.
   State that the slot-0 schedule, slot-0/3 contact and full-horizon crossing
   records, validators, and paired reducer are implemented. Preserve the
   one-cell candidate gate, exact 11-policy pair coverage, required fresh
   replay/dashboard review, 14-category limitation, and disabled
   qualification.
4. **Legal team starts, row 21:** replace `none` with
   `phase4_start_profile.profile_applied_before_observation`. State that the
   protocol-bound profile is applied during full-match validation reset;
   legality is measured on actually applied world poses; paired outcomes are
   compiled by exact policy/category/seed and profile. State that no accepted
   held-out source-bound report/replay review exists and candidate
   executability remains fail-closed.

Add concise source citations adjacent to these rows (or a short note directly
after the table) using the paths above. Do not describe code-produced
outcomes as executed Phase 4 evidence. The header statement of zero executable
qualification cells, qualification-disabled flags, and all simulator/replay
limitations remain unchanged.

**Pass 2 acceptance:** final diff is limited to these four rows and their
directly supporting citations; every event name and status claim matches the
current source anchors above; no runtime behavior, other status rows, or
qualification gates change. Submit the exact commit for independent Pass 3
review. No simulator/device work is in scope.

## Separate follow-up tranche: execution status and three-pass dashboard

Do not combine execution-status or dashboard-ledger edits with the Pass 2 table
correction. After Pass 2 exact-head review approves and integrates the row
correction, open a separately planned and reviewed metadata-only tranche to:

- update `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md` with the accepted row
  correction and its runtime-evidence limits;
- add a three-pass provenance entry for the reviewed documentation change in
  `docs/MECO_THREE_PASS_PROVENANCE.json`;
- update the pinned manifest digest in `frc_defense/dashboard_api.py` and its
  corresponding metadata contract coverage; and
- obtain an exact-head independent review for that metadata/dashboard change.

This is a separate gate because it changes dashboard provenance and its pinned
digest. Do not claim the dashboard reflects the correction until that later
tranche is reviewed and integrated. This plan and the planned row correction
do not require a dashboard restart, API call, replay, simulator, GPU, training,
or private campaign data.
