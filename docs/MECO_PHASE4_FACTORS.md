# MECO Phase 4 adversarial factors

This extension turns the four previously open Phase 4 conditions into frozen
matrix factors. It is a static implementation and contract only; it does not
claim runtime evaluation, qualification, or a robustness result.

## Supported evaluation domain

Use `offense_vs_scripted_pressure` or `full_match_offense` with registered
pressure categories at levels 1–5. A factor condition evaluates only the
eligible level 1–5 categories from the requested matrix; level 0 is explicitly
unsupported because it has no scripted pressure participant. Mixed requests
retain their full no-factor baseline while the factor report lists and
evaluates only eligible categories. Each treated category and seed is paired
with its row from `architecture_matched_cv_k16`, using the same five
checkpoints and D0/D1/D2 controls. Condition IDs bind the factor config,
category subset, checkpoint hashes, source identity, and seed list.

The scripted pressure participant is slot 0 at every supported level. The
feint and abrupt policy-change treatments replace slot 0's registered script;
other pressure slots retain their registered behavior (including level 5's
slot 1). Factor treatments preserve category, seeded layout, start condition,
sensor profile, and evaluation family. Unsupported categories and reasons are
retained in `phase4_unsupported_categories`.

The runner omits Phase 4 factors for other scenarios/categories and records
them as unsupported in the matrix protocol. It does not silently claim
generalization from unrelated scenarios.

## Frozen treatments

| Factor | Runtime treatment | Factor telemetry |
|---|---|---|
| Feinting opponent | Slot 0 telegraphs a deterministic lateral movement, then reverses at physics tick 360. | Reversal tick, whether reached, and policy metrics. |
| Observation delay | Buffer the observation and sensor-track state for six 20 ms physics ticks (120 ms); controls and MECO policies consume the delayed state. | Configured delay and delayed-observation tick count. |
| Independent low confidence | Cap the learner slot-3 sensed FUEL `track_confidence` at 0.2 after the initial reset perception and each subsequent perception refresh. The capped track state is captured in delayed observation history. Position, velocity, age, track mask, sensor dropout, and noise remain unchanged. Untracked/stale confidence values and every other robot are untouched. | Requested cap, applied cap, cap refresh count, number of sensed confidence values reduced, current sensed-track count, and final mean confidence for sensed learner tracks. |
| Abrupt opponent policy change | Slot 0 follows an orbiting policy, then switches to direct pursuit at physics tick 360. | Switch tick, whether reached, and policy metrics. |

Opponent factors reuse the registered reactive-pressure setup. The factor
controls only the stated opponent behavior on slot 0; the paired baseline retains
the category's registered level-specific script and all other scripted slots.
Factor metrics are descriptive episode outcomes, not promotion gates. Promotion
still requires the separate held-out protocol and calibrated safety gates.

The independent low-confidence treatment is implemented at the curriculum
environment's perception-update boundary. Initial scenario reset uses that same
boundary, and delayed sensor frames are captured after the cap is applied.
Evaluation has not yet been run, so these fields describe runtime instrumentation
and do not establish a measured robustness result. The isolated implementation
and provenance review are approved; integrated-tree review and seeded simulator
evaluation remain pending.
