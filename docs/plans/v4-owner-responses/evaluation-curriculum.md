# V4 evaluation and curriculum owner response

- **Owner role:** `evaluation_curriculum_owner`
- **Delegated task identity:** `/root/evaluation_curriculum_owner`
- **Response author:** delegated technical owner agent for this scoped row
- **Response date:** 2026-10-26
- **Disposition:** **ADOPT** the exact pinned V4 proposal for this owner row.

## Pinned artifacts

- V4 proposal: commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256
  `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- V4 owner-adoption matrix: commit
  `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256
  `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.
- Owner closure plan: commit
  `4bef4525acf8022f3b38edaafe0793d13ef7a0a2`, SHA-256
  `b0a72285269189f383d25e3463a258fa97a692bdee7674792e213eb0ec8657b7`.

This adoption is limited to the evaluation/curriculum owner row and the
semantics below. User-supplied owner decisions are recorded as inputs: the
primary simulated score estimand is accepted active-HUB FUEL score awards
only, explicitly not official match points; the PPO strategic reward target is
a fixed horizon, uses gross simulator reward, and keeps route/risk costs
separate; the current counterfactual defaults are retained. This response does
not claim that a Phase 0 approval artifact, V4 implementation, or qualification
evidence has been materialized.

## Adopted reward and counterfactual target semantics

1. **User-selected PPO strategic objective:** use a fixed 160-physics-tick
   strategic horizon, gross simulator reward, and keep route/risk costs
   separate. Current on-policy transition reward is the sum of
   `curriculum_metrics.reward_delta()` over physics ticks actually executed in
   each macro (`frc_defense/meco_track_training.py` lines 1915–1995, 1992; row
   capture at lines 2697–2707). A configured macro is at most 12 physics ticks;
   terminal, timeout or final partial macros may execute fewer
   (`MecoTrainConfig.action_ticks` default, lines 116–135). Current PPO GAE uses
   gamma `0.993` once per macro transition and lambda `0.95` once per transition
   (`_gae` and call site, lines 2031–2040 and 2151–2155). Its row reward is
   gross: route and risk costs are not subtracted (`_run_macro_step`, lines
   1990–1995; candidate route/risk evidence is carried separately at lines
   2686–2688).
2. **Counterfactual target defaults:** use `horizon_physics_ticks
   = 160`, `max_macro_ticks = 12`, `candidate_k = 16`, prediction mode `cv`,
   and `gamma_per_macro = 0.993`, matching
   `collect_counterfactual_v2` defaults (`frc_defense/meco_counterfactual.py`
   lines 1165–1170). Sum each executed macro's gross simulator reward without
   deducting route/risk cost. Preserve both gross undiscounted horizon return
   and the discounted horizon return. The first/selected branch is exponent
   zero; each continuation macro increments the exponent once regardless of
   macro duration, including a final partial macro (`meco_counterfactual.py`
   lines 70–73, 1132–1153). The branch uses the deterministic teacher
   continuation after the candidate action; it is a bounded rollout target,
   not global optimality (`meco_counterfactual.py` lines 1136–1153 and
   `docs/MECO_TRACKS_IMPLEMENTATION_STATUS.md` lines 557–569, 586–600).
3. **Separate route/risk costs:** route cost is ETA seconds multiplied by
   `0.05` simulated score points per second
   (`frc_defense/meco_candidates.py` line 34 and lines 732–740, 838–840).
   Risk is a distinct simulated-score-point penalty between zero and `0.75`
   points per forecast endpoint, linearly decreasing with separation inside a
   2.0 m threat radius and zero outside; multi-model mode takes the weighted
   hypothesis expectation (`frc_defense/meco_prediction_contract.py` lines
   12–23; `meco_candidates.py` lines 515–572). D1/D2 subtract route and risk
   separately from hand-written utility (`frc_defense/meco_track_training.py`
   lines 2885–2928). These are comparator/action-selection costs, not
   deductions from PPO's gross reward or counterfactual gross target. Values
   named “points” here are simulator utility units, not FRC official points.

The user-selected 160-tick strategic target and the current on-policy PPO GAE
implementation are not yet the same computation. The current implementation
uses per-macro reward and GAE through its scenario episode; it does not cap the
on-policy value target at 160 physics ticks. The counterfactual collector does
produce a fixed 160-tick discounted return, using deterministic teacher
continuation after the candidate branch. Therefore the adopted product choice
is fixed-horizon PPO reward semantics, but current source evidence only
implements those defaults for the bounded counterfactual target. A source plan
must close this gap before claiming PPO implementation conformance; it must
specify continuation policy for on-policy 160-tick targets and bind it to V4's
`return`/`advantage` fields. Do not silently call the teacher-continuation
counterfactual label the PPO return.

## Terminal, timeout, and truncation semantics

For an adopted V4 transition, a true task terminal does not bootstrap; an
execution-failure row is not policy-gradient credit; truncation bootstraps only
when a valid next observation is available. These are the V4 proposed
transition rules (`gpu-shared-command-ppo-transition-abi-v4-20261026.md`
lines 281–286) and align with the user-directed fixed target horizon. For a
fixed-horizon counterfactual target, stop at the first true terminal, scenario
timeout, or the 160-physics-tick target boundary; do not add a value bootstrap
to the finite target. Record the ending cause distinctly so a physical terminal
cannot be confused with horizon censoring.

There is an implementation discrepancy that remains a blocker for claiming
source conformance: current PPO rollout rows set `done = ended or timeout`
(`meco_track_training.py` lines 2697–2704), `_gae` uses that one `done` flag to
mask the next value (lines 2031–2040), and bootstrap is set to zero whenever
the last row is `done` (lines 2749–2752). Thus current training treats timeout
and terminal alike rather than bootstrapping a valid truncation observation.
The counterfactual collector also returns a combined `terminal` boolean for
physical end or timeout (`meco_counterfactual.py` lines 1136–1154), and needs a
typed end-cause field if it is to distinguish those from the fixed 160-tick
bound. V4 adoption specifies the required contract; it does not claim that
these source paths already implement it. This discrepancy crosses into the
PPO/RNG owner's row and requires reconciliation before ABI publication or
implementation planning.

## Primary simulated-score estimand

Adopt the explicit user decision that the evaluation estimand is the limited
simulated active-HUB FUEL award count, represented as
`simulated_match_score` in `simulated_score_unit`. The implemented ledger
counts accepted active-HUB FUEL awards only. Tower awards, fouls,
disqualification, ranking points, end-game bonuses and advancement points are
excluded; `official_match_points` remains null
(`docs/implementation-provenance/phase0-simulated-score-ledger-replay.md`
lines 21–31). This is an outcome separate from training reward, raw FUEL
piece-count diagnostics, useful throughput, and per-robot causal contribution.

The score-ledger implementation requires a source/rules-bound immutable
`meco-simulated-score-estimand-review-v1` that records the primary outcome and
unit, inclusion/exclusion rules, source/rules digests, intended comparison,
limits, approver and timestamp (`docs/plans/phase0-score-ledger-replay-implementation-20261025.md`
lines 42–50). The latest provenance says no approved artifact is present and
outcome collection remains blocked (`docs/implementation-provenance/phase0-simulated-score-ledger-replay.md`
lines 56–63). The user's decision resolves the product choice in this response,
but that exact immutable artifact and its Phase 0 digest binding remain a
runtime/evaluation prerequisite. Do not call this limited estimand official
match points.

## Qualification protocol, criteria, and artifact bindings

The relevant five-track qualification identity is the target-scoped calibrated
v5 protocol with implementation ID
`meco-v4-cluster-episode-device-percentile-v1`; it freezes one selected matched
reference (`D1/current`, `D2/cv`, or `D2/multi_model_v1`), keeps D0 as a separate
safety floor, and binds target, source/runtime, scenario closure, seed
partitions, predictor, safety schema and margins
(`docs/MECO_QUALIFICATION.md` lines 5–15, 225–249). A validation round requires
at least 100 episodes per category and frozen layout/family stratum, the same
protocol/checkpoint bundle for the candidate and all transitive prerequisites,
paired candidate/reference seeds, and separate D0 safety checks (same file,
lines 359–369). Promotion requires three consecutive passing target rounds
under that protocol; a failure or protocol change resets the streak, and the
third round's checkpoint bundle binds final-test evaluation (lines 388–396).

The campaign manifest binds `protocol_id`, `source_identity_sha256`,
`config_sha256`, comparator and safety-floor policy, validation/final-test seed
partitions, common ancestry, roots and Phase 0 identity. Each track checkpoint
manifest binds `policy_id`, declared `checkpoint_sha256`, source/config/
protocol/scenario, seed-partition digests and ancestry
(`docs/MECO_QUALIFICATION.md` lines 275–309). Preflight `ready: true` means
metadata consistency only: checkpoint bytes and outcomes are not verified and
no track is qualified (lines 322–357). Diagnostic campaigns cannot satisfy
this protocol.

`docs/TRAINING_CURRICULUM.md` is a broader proposed task curriculum, not the
operative five-track promotion protocol. It requires physical outcomes rather
than training return or requested actions, and its thresholds and scenarios
remain proposals until references, budgets and categories are frozen
(`docs/TRAINING_CURRICULUM.md` lines 5–7, 39–58, 101–132). Its 90% primitive,
cycle-throughput and retention proposals must not silently replace the v5
protocol's reviewed category predicates and margins. The v5 protocol and
Phase 0 evidence must explicitly bind the limited simulated-score estimand
alongside throughput/safety metrics before it can be a qualification gate.

## Optional Plan C recall diagnostic

Recall remains optional and separate from promotion. Its reference is the
deterministic, independent expanded-goal pool for currently sensed tracks,
bounded by route-qualified candidate-generation rules; it is not a
simulator-optimality oracle (`frc_defense/meco_candidates.py` lines 1670–1701).
The numerator is distinct selected, valid, route-feasible candidate identities
in that reference; the denominator is distinct route-feasible identities in
the reference pool. Report reference capacity and actual denominator. Current
source returns `1.0` on an empty reference; retain the zero denominator/empty
flag so that vacuous recall is not presented as measured coverage (same file,
lines 1678–1701). Summarize by track, category and K; no full-action recall or
simulator-optimality inference is allowed (`docs/MECO_ABLATION_MATRIX.md` lines
241–248). These recall fields do not replace reward or score estimands.

## Remaining blockers and dependencies

This row adopts the proposed V4 evaluation/curriculum boundary, with the
decisions above. The following exact evidence is still required before
qualification or shared ABI publication:

1. The immutable source/rules-bound human score-estimand artifact and its
   Phase 0 evidence-index/protocol identity binding. The user decision is
   received, but the required artifact is absent in current provenance.
2. Source correction and separate PPO/RNG owner reconciliation for terminal
   versus truncation flags, GAE/bootstrap behavior, and counterfactual end-cause
   serialization as described above. This response does not edit source.
3. Complete frozen Phase 0 capture/freeze/partition, D0 and matched-reference
   calibration, reviewed operational margins, v5 protocol, and five current
   track checkpoint/campaign manifests. `docs/implementation-provenance/phase0-baseline-freeze-contracts-20261025.md`
   lines 59–67 records that production freeze/partition/calibration/margin and
   match-point gate evidence is absent; `docs/MECO_QUALIFICATION.md` lines
   275–331 states the exact readiness chain and diagnostic-campaign block.
4. Matched held-out qualification and three passing rounds per track, with
   each result bound to the same frozen protocol and track checkpoint
   manifests. Static owner adoption, metadata preflight, or diagnostic
   evaluation is insufficient.

Cross-row dependencies are environment/profile and observation/native for
scenario/tick/identity; gamepiece/score for score-event coverage and lifecycle;
PPO/RNG for reward and terminal/bootstrap serialization; artifact/manifest
for approval, protocol and checkpoint digests; and Plan C/route for recall
identities and capacities. Any change to these dependencies that alters
fields, event completeness, target values or protocol identity requires the
affected owner rows to reconsider their decisions under the closure plan.

This is only the delegated evaluation/curriculum technical decision for this
row. It is not separate human maintainer approval, adoption by other owners,
shared ABI publication, or authorization for implementation, tests, runtime,
simulator/dashboard work, campaign changes, or training.
