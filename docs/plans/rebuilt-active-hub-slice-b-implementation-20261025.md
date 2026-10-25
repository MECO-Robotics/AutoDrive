# REBUILT active-HUB schedule and FUEL proxy: Slice B implementation plan

**Pass:** planner only
**Plan base commit:** `f17a1c7c89da00425abc60e0b5b28ee20498e704`
**Status:** proposed implementation sequence; no code, tests, simulator, replay, dashboard, training, or device work was run for this plan.
**Scope:** implement the pinned active-HUB schedule and an explicitly diagnostic `rebuilt_active_hub_fuel_proxy_v1` event seam for the MECO evaluation runtime. This plan does not produce official match points or qualification evidence.

## Source pins and authority

This plan is bound to the following repository inputs as they existed at the plan base:

| Input | SHA-256 |
|---|---|
| `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md` | `718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10` |
| `docs/plans/rebuilt-match-points-ledger-20261025.md` | `05f11b1838643258ac57f56734e1ee33aacd511230e59dcef88942cc1713d1c9` |
| `qualifications/meco/rebuilt-rules-TU22.json` | `18cf92041290e8fa1df15d7e967fe5e107118a6d443fe2b6526c1c05b73f47a8` |

The scoring contract pins the official 2026 REBUILT Game Manual TU22 (`5c67300fc412a1eed8ca9dc37fb6644aee05795a2befed1f69736643cbf95139`) and combined Team Updates through TU22 (`8ec2754d157b7f35cf93b4be301ebdf576f2abf2650975731dcc3cdf5ab0ecb1`). Recheck these source and rules-snapshot digests before implementation. If any changes, stop and revise the plan against the new pinned contract; do not silently update a digest.

The plan follows the evaluation document's five shared tracks (A1, A2, A3, C1, C2), common deterministic safety/planning boundary, matched evaluation, and metric gates. A shared proxy must use one contract and one unit across all five tracks. The evaluation document is an experimental specification, not qualification authority.

## Runtime trace and correction to Slice B's file list

The MECO evaluation path is:

1. `frc_defense/meco_track_training.py::evaluate_meco` creates an evaluation episode and calls `make_curriculum_env`.
2. `frc_defense/curriculum_scenarios.py::make_curriculum_env` constructs `CurriculumEnv`; `CurriculumEnv` subclasses `TensorThreeVsThreeEnv`.
3. `frc_defense/tensor_3v3.py::TensorThreeVsThreeEnv.step` advances the shared physical simulator, then calls `_update_match(active)`, followed by `_update_fuel(active, score_intent=...)`.
4. `frc_defense/_tensor_3v3_gamepieces.py::TensorThreeVsThreeGamepieceMixin._update_match` owns match elapsed time and HUB state for this path. Its normal schedule currently uses 30–130 seconds for four shifts of 25 seconds, and both HUBs are active outside those shifts. It currently resolves AUTO ties by defaulting to RED; that does not encode the pinned FMS tie result.
5. `_update_fuel` calls `frc_defense/gamepiece_actions.py::update_fuel_actions`, which emits per-robot `fuel_scored_event` counts and increments the raw alliance `fuel_score_count`. `frc_defense/_tensor_3v3_actions.py` also reads `hub_active` while selecting/committing actions.
6. `frc_defense/tensor_3v3.py` imports `TensorVectorizedSimulator` from `tensor_sim.py`; that is the shared physics dependency, not the owner of the REBUILT clock.

The existing Slice B draft names `tensor_defense_gamepieces.py`, `tensor_defense_reset.py`, and `tensor_defense_step.py`. Those belong to the separate two-robot `TensorDefenseEnv` path. They are not the path created by `evaluate_meco`. Do not implement Slice B there as a substitute for the MECO runtime. Update the Slice B source list to the actual files below; touch the legacy two-robot path only if a separately scoped compatibility requirement is approved and tested.

There is a second prerequisite in Slice A's current contract: `frc_defense/rebuilt_match_scoring.py` stamps the proxy scope as `two_robot_2d`, even though the MECO path constructs six simulator slots. Do not route a proxy-only MECO record through `build_score_ledger` as it currently exists: that function first validates official alliance rosters, active lineups, and six official team/robot slots, which would force fabricated or inferred team identity into a restricted proxy. Add a separate pure proxy-only validator/aggregator (for example, `build_fuel_proxy_ledger`) that accepts only a simulation episode identity, the pinned rules digest, explicit simulator slot-to-alliance mapping, deterministic match times, and abstract score-action rows. It must not require `team_id`, alliance roster, official lineup, competition context, or official FMS record; it must not return official context or official score. If stable slot IDs are needed, use plainly synthetic identities such as `sim_red_slot_0`, bound to scenario/seed/episode and marked `simulation_only=true`; never represent them as FRC team IDs. Explicitly keep this proxy-only schema/API disjoint from `build_score_ledger` and `official_match_points`.

Before Slice B serializes a scope, revise Slice A's proxy contract/tests so the label reflects six simulator slots in a 2D simulated environment, not `two_robot_2d`, while retaining every non-qualification flag. This scope correction is not permission to claim six official active-match robots: the slots still lack sensor-verified FUEL, Tower adjudication, referee inputs, official roster identity, and other complete-score evidence.

## Existing behavior and implementation decisions

The active `TensorThreeVsThreeGamepieceMixin._update_match` already has the nominal official shift boundaries in its normal path: AUTO and TRANSITION precede shifts at 30, 55, 80, and 105 seconds; both HUBs are active from 130 seconds. Slice B therefore must not blindly duplicate the already-correct schedule. It must make the effective contract exact and auditable, fix the tie path, and test the state/action seam end to end.

The implementation must:

- Use `rebuilt_match_scoring.py` and the pinned rules snapshot as the single source for half-open phase and HUB schedule semantics: AUTO `[0,20)`, TRANSITION `[20,30)`, SHIFT 1 `[30,55)`, SHIFT 2 `[55,80)`, SHIFT 3 `[80,105)`, SHIFT 4 `[105,130)`, END GAME `[130,160)`.
- Preserve the existing step order unless evidence shows it violates the rule contract. Document that the clock is advanced by one `dt` before the current tick's score action is resolved, then ensure boundary tests assert which half-open interval owns the score event at an exact simulated timestamp. Do not hide an off-by-one behind float tolerances.
- Define the SHIFT 1 leader/tie input from score events whose event time is exactly in the half-open AUTO interval `[0,20)` and no other events. At the 20-second transition boundary, freeze that count pair and its derived leader/tie input before any TRANSITION scoring can occur; the frozen value is immutable for all later shifts in that episode. TRANSITION events in `[20,30)` must never change the SHIFT 1 choice.
- The current scorer increments `auto_fuel_scores` when `match_elapsed <= 20 + dt`, which can include TRANSITION events due to the clock advancing before scoring. Change its gate to strict AUTO event time `[0,20)` (for example, use the score event's recorded match timestamp `<20.0`) and derive/freeze the schedule input only from those same qualifying source events. Keep the total raw `fuel_score_count` semantics unchanged. Add an explicit regression proving a score event at exactly `20.0` and one at `20.0 + dt` do not enter AUTO totals or affect the leader/tie, while an event just below `20.0` does.
- For this restricted proxy only, resolve AUTO ties with an explicit deterministic **simulation tie selection** bound to the episode seed/match identity; it is a reproducible stand-in, not an actual FMS result. Bind the value and derivation in the proxy record and use an isolated deterministic derivation or generator so it does not perturb physics/policy RNG streams. The proxy must remain ineligible for official scoring because it lacks an actual FMS record. Non-tied AUTO totals select the manual-defined higher-AUTO alliance as SHIFT 1 inactive. Alternate the inactive alliance by shift. Both HUBs remain active during AUTO, TRANSITION, and END GAME.
- Preserve explicit `curriculum_schedule` and `behavior_probe` overrides as test/diagnostic controls, but make their override status visible in the emitted record and prevent them from being presented as a rules-compliant match schedule. The production MECO path must not accidentally inherit either override.
- Keep `fuel_score_count` semantics and existing reward units unchanged. Correct `auto_fuel_scores` to count only score events with event time in `[0,20)`; record that semantics correction in source/tests. Do not rename, reinterpret, or add proxy values into `fuel_score_count`.
- Create a per-world, ordered proxy event ledger from actual `fuel_scored_event` emissions at the common `update_fuel_actions` seam. Each row must carry a stable unique event ID, source event/tick, match time, alliance, robot slot where known, `event_type=fuel_action_score`, and `event_semantics=abstract_score_action_not_sensor_crossing`. Event IDs must be deterministic and duplicate-rejected. Preserve per-event rows instead of reconstructing them from final count deltas.
- Run rows through the new proxy-only pure validator/aggregator, not the official-context `build_score_ledger` API. Its inputs and output must be simulation-only and must not require or infer FRC team IDs or official roster/lineup/context. Keep the output under `rebuilt_active_hub_fuel_proxy_v1`; never add a generic `match_points` field. Explicitly state that official score is unavailable (`official_match_points=null` only in a separate enclosing report if needed), `qualification_eligible=false`, and Tower, referee, official roster, sensor-crossing, score-window, and official context completeness are unknown/incomplete. Do not imply missing score categories equal zero.
- Do not manufacture FUEL sensor crossings, 3-second processing/grace events, Tower scores, foul/no-foul decisions, or per-team attribution that the runtime does not observe. Current score-action events remain a legacy abstract diagnostic submetric. The proxy may value only the permitted abstract event rows under the pinned effective HUB schedule and must describe the missing sensor timing/assessment windows.
- Do not alter the running r4 checkout, service, checkpoints, reward, manifests, queue, evaluation cadence, or device leases. The code change must be passive with respect to the active campaign; a future fresh five-track campaign can bind the common proxy contract only after the user stops r4 or it ends and both WX9100 leases are free.

## Planned implementation surface

Primary runtime files:

- `frc_defense/_tensor_3v3_gamepieces.py`: centralize/derive the effective phase and HUB schedule from the pinned pure contract; replace the implicit RED tie fallback with explicit simulation-only seeded tie state; freeze the qualifying AUTO count pair at 20.0 before transition score handling; expose frozen schedule input, effective state, and override provenance needed for per-world events.
- `frc_defense/tensor_3v3.py`: initialize/reset AUTO count, frozen schedule input, tie selection, and proxy-event state per world; ensure partial resets clear only selected worlds; retain six simulator-slot attribution and raw `fuel_score_count`; establish deterministic simulation-only episode identity/seed binding. Only change step ordering if the exact-boundary tests prove it necessary.
- `frc_defense/gamepiece_actions.py`: at the shared FUEL action score event producer, append stable proxy source rows for actual per-robot score events. Preserve existing `fuel_scored_event` count tensors and metrics. Respect active masks, partial worlds, event ordering, and GPU capture/kernel constraints.
- `frc_defense/curriculum_scenarios.py`: pass episode identity/tie derivation to the environment through an explicit parameter rather than hidden global state; ensure production construction has no curriculum/probe HUB override.
- `frc_defense/rebuilt_match_scoring.py` and `qualifications/meco/rebuilt-rules-TU22.json`: add the separate proxy-only API and accurate simulation scope; do not make it call official `_validate_context`. Update the snapshot digest constant and contract tests atomically if the JSON bytes change. Preserve the pinned official document hashes.
- `frc_defense/meco_track_training.py`: expose proxy ledger and separate contract/digest fields in returned evaluation records; keep `fuel_score_count` raw; record actual six environment slots without inferring official active roster identity; mark the result diagnostic and non-qualifying.
- `frc_defense/tensor_training_ppo.py` and/or existing common training/evaluation seam: if the shared proxy reward/metric is enabled for training, make it explicitly opt-in and manifest-bound across all five tracks. Do not change r4's in-flight objective. Do not add a track-specific reward variant.

Do not modify the legacy `tensor_defense_*` modules for this MECO change unless a new call-path audit proves they are used by `evaluate_meco`; at current plan base, they are not.

## Invariants and focused tests

Add or extend tests in `tests/test_tensor_3v3.py`, `tests/test_fuel_physics_integration.py`, `tests/test_rebuilt_match_scoring.py`, and the MECO replay/evaluation tests as needed. Keep pure contract tests separate from simulator behavior tests.

Required exact clock/state coverage:

- Immediately before, exactly at, and immediately after 20, 30, 55, 80, 105, 130, and 160 seconds. Tied AUTO cases use the record's simulation-only seeded tie result; never label it as an actual FMS decision.
- AUTO schedule input counts only scorer events timestamped in `[0,20)`; freeze the counts and leader/tie result at 20.0 before TRANSITION. Prove events at `20.0` and `20.0 + dt` cannot affect either AUTO count or SHIFT 1, despite the previous `<=20+dt` behavior; prove an event just below 20.0 does count.
- Both HUBs active in AUTO, TRANSITION, END GAME; exactly one inactive in each shift; alternation direction and 25-second duration; non-tied AUTO selection; tied proxy cases with both possible seeded simulation tie selections. Separately, pure official-ledger tests continue to require an actual explicit FMS selection for tied official AUTO events.
- `active_mask` false freezes clock, HUB status, tie state, score-event counters, and per-world ledger; partial reset reinitializes only selected worlds and does not reuse old event IDs.
- Curriculum/probe overrides remain isolated and are labeled non-production. A production proxy episode lacking a seeded simulation tie selection must fail closed on AUTO tie; it must not silently pick RED or claim an actual FMS selection.
- At the scorer seam, actual active-HUB action events create exactly one immutable proxy row per FUEL score, with deterministic ordering/IDs and source attribution. Inactive-HUB events cannot score. Raw `fuel_score_count`, `auto_fuel_scores`, and rewards retain their prior meaning.
- Proxy-only validation accepts no team ID, roster, lineup, competition context, or official FMS identity; it validates only simulation identity/seed, synthetic slot-to-alliance mapping, pinned source/rules digest, event time/source, and deterministic tie selection. It rejects duplicate/malformed rows, stale scope, incorrect event time, or any accidental `match_points`/official-context field. A separate test proves official `build_score_ledger` still requires its full official context and cannot be called by the proxy-only path. Proxy values remain separate from official score and can never make a current MECO case qualification eligible.
- CPU tests cover semantic reference behavior. Device parity tests compare CPU and supported accelerator schedule tensors/event outputs for the same seeds and transitions, using the already-supported devices only; they do not require P600. Keep all six slots and multiple worlds in a case to expose team indexing/masking errors.
- No test may assert that action-intent events are sensor crossings or that absent Tower/referee inputs mean zero points/no fouls.

## Required end-to-end simulation and replay evidence

Because these tests change HUB/FUEL simulated behavior, focused unit tests are insufficient. After implementation and tests, run one reproducible end-to-end MECO simulator evaluation that exercises AUTO scoring, the AUTO tie-break path, both alliance shifts, and HUB reactivation in END GAME. Prefer a short deterministic registered `full_match_offense` or purpose-built registered scenario with a fixed scenario ID and fixed episode seed; the evaluator must execute the real `evaluate_meco` → `make_curriculum_env` → `CurriculumEnv` path. Capture the replay from the same fresh evaluation, not an older cached run. The replay must expose clock phase, active HUB state, AUTO/tie selection, raw FUEL events, proxy event IDs/digest, and explicit non-qualification flags.

Before reviewing that replay, follow the project `AGENTS.md` requirements:

1. Confirm the dashboard service is serving the current implementation checkout, then restart the relevant dashboard service(s) after code changes so Python modules reload.
2. Verify the service is active and dashboard endpoints respond successfully. At minimum inspect `/api/curriculum-status` and `/api/meco-status`; record response status, source/currentness metadata, and any non-200 or stale evidence.
3. Confirm the served frontend matches the current checkout: fetch the dashboard HTML and every referenced JS/CSS asset from the service, compute hashes, and compare them with the corresponding current-checkout files. Record the exact asset paths and hashes. Do not rely only on an HTTP 200 or backend version.
4. Generate the replay only after the restart/currentness checks. Record scenario ID, episode seed, source commit/tree and protocol/rules digests, replay/simulation ID, output path, and the result of inspecting the replay frames/record around all clock boundaries and score events.
5. If the existing dashboard cannot render an updated replay schema, fix the dashboard/replay integration before declaring the simulator check complete. Mark any replay produced by the old source/service as outdated.

The plan artifact itself produces no simulation evidence. The implementation report must state the exact tests, simulation scenario and seed, replay identifier, dashboard service/API status, frontend hashes, proxy result, and that no qualification claim was made. If current r4 training or device leases prevent a required hardware-backed parity run, record that blocker and complete CPU-only plus end-to-end simulator work where supported; never pause or displace r4 to create test capacity. Hardware parity cannot be silently waived if changed device state requires it.

## Planner acceptance and implementation review gates

1. Independent reviewer confirms the call-path trace, scope correction, source pins, exact boundary semantics, no synthetic authoritative evidence, and compatibility with Slice A.
2. Implementer starts from a fresh worktree at the then-current pushed main, updates this plan if relevant source/document hashes changed, and makes small coherent commits pushed to its branch.
3. Independent reviewer reviews the exact implementation commit(s), including tests and generated simulator evidence; changes after review require a new exact-head review.
4. Integrate only reviewed commits. Restart/check the dashboard after integration and bind the new source tree/protocol digest in any subsequent campaign.
5. This slice cannot pass the overall plan's official match-points gate. Slice D still requires complete roster, sensor-equivalent FUEL events and scoring windows, Tower adjudications, referee/penalty and DQ context, and separate official result handling. No `meco-qualification-v5` or official `match_points` may be emitted by Slice B.
