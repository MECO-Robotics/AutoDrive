# V4 evaluation and curriculum owner response

- **Owner role:** `evaluation_curriculum_owner`
- **Delegated task identity:** `/root/evaluation_curriculum_owner`
- **Response author:** delegated technical owner agent for this scoped row
- **Response date:** 2026-10-26
- **Disposition:** **PENDING**

## Pinned artifacts

- V4 proposal: commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256
  `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- V4 owner-adoption matrix: commit
  `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256
  `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.
- Owner closure plan: commit
  `4bef4525acf8022f3b38edaafe0793d13ef7a0a2`, SHA-256
  `b0a72285269189f383d25e3463a258fa97a692bdee7674792e213eb0ec8657b7`.

## Evaluation identity and qualification gate

The relevant five-track promotion path is the calibrated target-scoped v5
qualification protocol, implementation ID
`meco-v4-cluster-episode-device-percentile-v1`. The protocol freezes one
matched reference (`D1/current`, `D2/cv`, or `D2/multi_model_v1`), separately
reports D0 as the safety floor, and is bound to the target, source/runtime,
scenario closure, seed partitions, predictor contract, safety schema and
reviewed margins (`docs/MECO_QUALIFICATION.md` lines 5–15, 225–249). Each
round requires at least 100 episodes per category and frozen layout/family
stratum, the same protocol and checkpoint bundle for the learned policy and
all transitive prerequisites, paired learned/reference seeds, and a separate
D0 safety comparison (same file, lines 359–369). Promotion requires three
consecutive passing target rounds under that frozen protocol; a failure or
protocol change resets the streak, and the third passing checkpoint bundle is
the one bound to final-test evaluation (lines 388–396).

The Stage 1 manifests bind `protocol_id`, `source_identity_sha256`,
`config_sha256`, comparator and safety-floor policy, validation/final-test seed
partitions, common campaign ancestry, roots, and Phase 0 identity. Each
track-specific checkpoint manifest binds `policy_id`, declared
`checkpoint_sha256`, source/config/protocol/scenario, seed-partition digests,
and campaign ancestry (`docs/MECO_QUALIFICATION.md` lines 275–309). The
readiness preflight is metadata-consistency evidence only; it does not read or
verify checkpoint bytes, execute a model, read validation outcomes, or qualify
a track (lines 322–357). Diagnostic campaign records remain ineligible.

These qualification gates are distinct from the broad task curriculum in
`docs/TRAINING_CURRICULUM.md`. That document labels its scenarios and thresholds
as proposals, calls its infrastructure unqualified, and says promotion must
use physical outcomes rather than accumulated training reward or requested
action counts (lines 5–7, 39–58, 101–130). Its primitive-task proposal is at
least 90% success in each required category across three validation
evaluations; cycle and pressure stages add separate useful-throughput and
retention predicates (lines 120–132). Those broad proposals are not silently
substituted for the current five-track v5 qualification protocol. The exact
approved target scenario and category/layout coverage must be frozen in the
protocol and Phase 0 artifacts before qualification.

## Score estimand and training reward

I cannot adopt V4 yet because required product choices are not closed by
source facts. V4 records `reward:float32`, `macro_reward:float32[12]`,
`terminal`, and `truncated`, and says true terminal/execution failure do not
bootstrap while truncation bootstraps only with a valid next observation. It
explicitly leaves the product estimand to the curriculum/simulator owners
(V4 proposal lines 147–160 and 281–286). The learning-track plan asks C1/C2 to
start from gross strategic reward with downstream deterministic costs applied
once, but leaves fixed-horizon versus next-tactical-decision reward unresolved
and requires consistent discounting (`docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`
lines 49–57, 63–67, 102–104, 150–158). Plan A's sampled supervised U target
and Plan C's sampled counterfactual regret already have their own bounded,
discounted-return contracts; they are not evidence of one shared PPO reward
estimand (`docs/MECO_TRACKS_IMPLEMENTATION_STATUS.md` lines 528–539, 557–569,
586–600).

Phase 0 defines a separate simulated score-ledger candidate outcome:
`simulated_match_score` in `simulated_score_unit`, counting accepted active-HUB
FUEL score awards only. It excludes tower awards, fouls, disqualification,
ranking points, end-game bonuses and advancement points; official points remain
null (`docs/implementation-provenance/phase0-simulated-score-ledger-replay.md`
lines 21–31). This is not the training reward, raw FUEL count, useful
throughput, official match points, or robot-attributed causal contribution.
The human-authored `meco-simulated-score-estimand-review-v1` approval is a hard
prerequisite to outcome collection. The provenance record says no approved
artifact exists and collection remains blocked (same file, lines 56–63).
The implementation contract requires the human decision to name the primary
outcome/formula/unit, included and excluded scoring/terminal effects,
interpretation limits, intended comparison, source/rules digests, approver and
time; an out-of-contract event makes the episode incomplete
(`docs/plans/phase0-score-ledger-replay-implementation-20261025.md` lines
42–50). The baseline-freeze summary likewise says no production freeze,
partition, calibration, margin review or match-point gate exists and current
score semantics remain scored-FUEL piece counts (`docs/implementation-provenance/phase0-baseline-freeze-contracts-20261025.md`
lines 59–67).

The following choices therefore remain explicit product-owner gates, not
inferences made by this delegated response:

1. **PPO reward target:** choose fixed strategic horizon or next tactical
   decision; define tick/macro aggregation, discount and risk-penalty units,
   and say how terminal and truncation affect the target. Keep gross reward and
   downstream route/risk costs from being double-counted, consistent with the
   learning-track plan.
2. **Primary qualification estimand:** the human Phase 0 approver must choose
   and bind the primary simulated score outcome and its intended use. Resolve
   whether it is the limited active-HUB simulated score ledger as currently
   implemented or a broader scope requiring additional event coverage. Do not
   label the limited proxy official match points.
3. **Qualification predicate binding:** freeze the v5 target, comparator,
   protocol identity, category/layout/family rows, margins and pass predicates
   in the reviewed Phase 0 evidence and protocol; each A1/A2/A3/C1/C2 checkpoint
   bundle must carry the exact manifest bindings above. The broad curriculum's
   proposed 90% or useful-throughput targets cannot become operational gates
   without their required frozen references and approved protocol.

Until these gates are resolved and represented by the human approval plus
source-bound Phase 0/protocol artifacts, a training reward, a diagnostic run,
or metadata-ready preflight cannot count as qualification. Therefore this row
remains **PENDING**; I do not create a replacement ABI or choose product
estimand values.

## Optional Plan C recall diagnostic

Candidate recall is a separate diagnostic and not a promotion predicate. The
source's precise reference is an independent, deterministic expanded-goal pool
over currently sensed tracks, bounded by candidate-generation rules; it is not
a simulator-optimality oracle (`frc_defense/meco_candidates.py` lines
1670–1701). Define the numerator as the count of distinct selected valid,
route-feasible candidate identities in that reference pool and denominator as
the count of distinct route-feasible identities in the reference pool. Report
the pool capacity and actual denominator. Current source returns `1.0` when
the denominator is zero; preserve the zero denominator/empty-reference flag so
that vacuous empty-pool value is never presented as measured coverage (same
file, lines 1678–1701). Summarize by track, category, and K; the ablation
contract explicitly prohibits inferring recall over the full action space or
simulator optimality (`docs/MECO_ABLATION_MATRIX.md` lines 241–248). If V4
retains recall fields, these diagnostics must remain separate from reward,
score estimand and qualification outcomes.

## Cross-row dependencies and limits

This row depends on environment/profile and observation/native for the frozen
scenario, simulator ticks, target/episode identity and terminal event context;
gamepiece/score for score-event coverage, piece lifecycle and score rules;
PPO/RNG for reward, terminal/bootstrap and truncation semantics; artifact/
manifest for immutable protocol, source, checkpoint and estimand-approval
bindings; and Plan C/route for the bounded recall reference identities and
capacity. A change to any of these may change score completeness, episode
eligibility, payload fields, protocol identity, or the meaning of a reported
metric. This response does not adopt those owners' fields. Any replacement or
changed dependency requires the affected rows to decide again under the owner
closure plan.

This records only the delegated evaluation/curriculum technical decision for
this row. It is not human maintainer approval, adoption by another owner,
cross-owner V4 adoption, ABI publication, or authorization for implementation,
tests, runtime, simulator evaluation, dashboard work, campaign changes, or
training.
