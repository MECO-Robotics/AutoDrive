# MECO Phase 4 adversarial factors

This extension turns the four previously open Phase 4 conditions into frozen
matrix factors. It is a static implementation and contract only; it does not
claim runtime evaluation, qualification, or a robustness result.

## Supported evaluation domain

Use `offense_vs_scripted_pressure` or `full_match_offense` with one or more
categories ending in `level_4_reactive`. Each matrix condition evaluates the
same five checkpoints and D0/D1/D2 controls against the same categories and
explicit seed list. The `architecture_matched_cv_k16` condition is the
no-factor paired baseline. Every condition ID includes the factor config,
checkpoint hashes, source identity, and seeds; the matrix report retains the
per-condition evaluation reports and paired episode metrics.

The runner omits Phase 4 factors for other scenarios/categories and records
them as unsupported in the matrix protocol. It does not silently claim
generalization from unrelated scenarios.

## Frozen treatments

| Factor | Runtime treatment | Factor telemetry |
|---|---|---|
| Feinting opponent | Slot 0 telegraphs a deterministic lateral movement, then reverses at physics tick 360. | Reversal tick, whether reached, and policy metrics. |
| Observation delay | Buffer the observation and sensor-track state for six 20 ms physics ticks (120 ms); controls and MECO policies consume the delayed state. | Configured delay and delayed-observation tick count. |
| Independent low confidence | Cap the learner's sensed FUEL `track_confidence` at 0.2. Position, velocity, age, track mask, sensor dropout, and noise remain unchanged. | Applied cap and final mean confidence for sensed learner tracks. |
| Abrupt opponent policy change | Slot 0 follows an orbiting policy, then switches to direct pursuit at physics tick 360. | Switch tick, whether reached, and policy metrics. |

Opponent factors reuse the registered reactive-pressure setup. The factor
controls only the stated opponent behavior; the paired baseline retains the
registered level 4 reactive pursuit. Factor metrics are descriptive episode
outcomes, not promotion gates. Promotion still requires the separate held-out
protocol and calibrated safety gates.
