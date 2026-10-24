# D0 replay cadence for control-only evaluation

## Planner summary

Control-only diagnostic D0 has no learned-policy `action_ticks` value, but its native controller advances one physics tick per `env.step`. Replay grouping therefore needs a D0-specific interval: use the configured action cadence for paired runs and one tick when that cadence is absent. Keep learned-policy cadence metadata unchanged, report native D0 `action_repeat=1`, and flush a short terminal group once.

## Implementer summary

Implementation commit `9cf17820e3034aaa1e270c5d071dc7289c96e637` added pure replay-cadence helpers. Control-only D0 maps unset cadence to one tick; paired cadence 12 remains 12. Decision windows, executed-action slices, terminal flush, and replay capture stride use the D0 replay interval, while `action_ticks` in evaluation metadata remains unset for D0-only runs. Candidate integration commit `08b93828032ca6561552358142bb6b1781bc5071` contains the change. The focused replay module passed 33 tests, with compilation and diff checks passing.

## Reviewer summary

The independent reviewer approved the exact implementation commit. Review confirmed one-tick D0-only frames, preserved 12-tick paired grouping, a single terminal flush, and unchanged evaluation cadence metadata. The reviewer did not rerun tests or simulator evaluation.

## Runtime evidence and limits

The integrated-candidate simulator evaluation completed one GPU D0 diagnostic episode for `pickup_simple/cluster`, seed `61720261024`, using `cuda:0` on an AMD Radeon Pro WX 9100. It produced 59 frames through tick 58 and a successful scenario completion after six FUEL acquisitions (1.16 simulated seconds). The current dashboard served the replay with source commit `08b938280`, no dirty files, and `outdated=false`. This does not establish full D0 calibration, D1/D2 comparison, a learned-track result, or qualification.
