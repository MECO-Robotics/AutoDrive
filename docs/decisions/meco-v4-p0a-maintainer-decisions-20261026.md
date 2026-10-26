# MECO V4 P0a maintainer decision record

**Decision status:** Approved by the authorized maintainer via the conversation reply `Approve all` on 2026-10-26. The exact reply timestamp is not exposed in the conversation export; this record is being prepared at 2026-10-26 13:31 UTC. No cryptographic signature is asserted.
**Approved candidate:** `docs/plans/v4-phase0-preregistration-candidate-20261026.md`, commit `56183cefd2402342ce28f836155164fc850ebc60`, file SHA-256 `c8ace83e0e5a3fdef8d33ab090d3cbb9193e12342b015d72b07016acdb603726`.
**Decision scope:** The maintainer approved the proposals presented with that exact P0a candidate: the source-defined score proxy, both finite case matrices, and the proposed analysis, safety, and failure rules. This records protocol decisions only. It does not authorize builds, tests, simulator runs, calibration, GPU use, dashboard operations, training changes, integration, merge, promotion, or stopping/restarting the active campaign.

## Approved score estimand and protocol roles

- The primary research outcome is `simulated_match_score`, the limited simulated proxy of accepted active-HUB FUEL awards. For each complete episode, sum `score_units` by alliance over unique authoritative accepted award event IDs and retain the red/blue totals. Do not claim per-robot causal contribution. This is not official match points; the explicitly excluded outcomes remain tower awards, foul penalties, disqualification, ranking points, end-game bonuses, and advancement points.
- Stage 1 qualification uses `pickup_simple` only, with the registered `single` and `cluster` categories and the curriculum acquisition/retention predicate. The primary score-evaluation protocol uses `full_match_offense` and its registered prerequisite closure. They are separately bound and cross-referenced; neither protocol's result substitutes for the other's outcome.
- The fixed reward contract remains the previously confirmed current counterfactual defaults, with a 160-physics-tick strategic horizon, at most 12 ticks per macro, gamma 0.993 per continuation decision, gross reward as the PPO target, and route/risk costs reported separately.

## Approved proposed case matrices

Stage 1 `pickup_simple` matrix, applied to both registered categories and paired across the five tracks and deterministic comparators:

| Partition | Layout | Defender | Start | Sensor | Episodes per category/stratum |
|---|---|---|---|---|---:|
| Train | `2026_rebuilt` | `none` | `scenario_defined` | `standard` | Existing training protocol |
| Calibration | `2026_rebuilt_shifted_structures_v1` | `none` | `offset_upper_lane` | `noisy` | 100 |
| Validation | `2026_rebuilt_shifted_structures_v2` | `none` | `offset_upper_lane` | `noisy` | 100 |
| Final test | `2026_rebuilt_shifted_structures_v3` | `none` | `offset_lower_lane` | `noisy` | 100 |

Primary score `full_match_offense` matrix uses the same partition-specific layout, start, and sensor rows. For the two pressure scenarios (`offense_vs_scripted_pressure` and `full_match_offense`), use `category_scripted` in train, `orbiting_interceptor` in calibration/validation, and held-out `split_chaser` in final test. Use `none` for non-pressure scenarios across the full registered prerequisite closure. The accepted v5 axis override uses one layout per partition and keeps the score protocol separate from Stage 1.

The proposed partition and sample rules are approved: disjoint train/calibration/validation/final-test membership; 100 episodes per category/stratum in calibration, validation, and final test; at least two calibration repetitions; deterministic seed construction and paired seeds as specified in the candidate. Exact expanded case/seed bytes and frozen opponent policy/checkpoint/config identities must be source-bound in the protocol before any new candidate training or outcome collection; approval of the matrix does not invent those identities.

## Approved analysis, safety, and failure proposals

- Use 10,000 paired-block bootstrap replicates, resampling paired case/seed blocks within scenario/opponent-family strata and retaining every policy row in each sampled block. Apply Bonferroni familywise 95% coverage across five track contrasts using two-sided 99% percentile intervals (0.5th and 99.5th percentiles). The selected comparator is chosen on the score-calibration partition only and then frozen.
- Report the source-defined `meco-safety-rates-v1` contact metrics separately: `safety_robot_contact_rate`, `safety_field_contact_rate`, and `safety_wall_contact_rate`, defined as focal-slot-3 contact ticks divided by active physics ticks. Use calibration-derived noninferiority margins. Contact rates do not substitute for hard safety events.
- Apply the proposed hard veto for any source-bound executed deterministic-legality violation, swept-footprint violation, capture/mechanism constraint violation, or failed required safe fallback. A remasked proposed action alone is diagnostic, not proof of an executed violation. No safety pass may be claimed until each veto has an implemented and reviewed source-bound signal.
- Keep every scheduled row; do not replace cases or paired blocks. A complete zero-award score episode is zero. Incomplete score episodes are non-evaluable and never zero-imputed. A Stage 1 timeout fails the round. `safety_stop` fails the safety gate. Simulator, worker/device, invalid/stale-record, and incomplete-episode dispositions follow the per-class table in the approved candidate, with original reasons retained and no unsupported safety pass.
- Compare A1/A2/A3/C1/C2 against the single calibration-selected strongest comparator from D0/D1/D2. D3 remains excluded unless a preregistered amendment is approved before candidate outcomes.

## Boundaries and provenance

This record approves the proposed protocol choices only. It does not certify that the score event producer, safety signals, or runner implementation are complete; it does not freeze numeric useful-gain or safety margins, which are derived from calibration-only evidence under the approved method. The P0a candidate and this record require independent exact-head Pass 3 review before P0a acceptance. P0b, all source implementation, runtime validation, and any new training campaign remain subject to the accepted plans, G0, and separate authorization. The existing training campaign is unaffected.

The conversation approval is the provenance for this record. Agent-generated hashes identify the candidate and record bytes but do not constitute a human signature.
