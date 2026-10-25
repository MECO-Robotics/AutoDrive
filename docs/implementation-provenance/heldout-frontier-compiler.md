# Held-out frontier checkpoint binding

## Planner

Section 6 of the learning-track plan requires a frozen, matched held-out
comparison before qualification. The frontier compiler must bind each of the
five supplied configuration hashes to its actual checkpoint payload, verify
the embedded track and common training protocol, and preserve the validation
only/final-test non-disclosure rules. It should refuse unsupported payloads
and must not describe an input lock as an evaluation result.

## Implementation

Commits `9841b85dd` and `af0a4f1f4` introduced the offline frontier compiler and
scenario-job coverage checks. Corrective commit
`39976d4aa64816facec21988e12da29bb07dcea5` now loads each supported schema-1
checkpoint, recomputes its immutable configuration digest with only `updates`
excluded, checks the embedded track, requires one shared nonempty training
protocol ID, and rejects caller hashes that do not match. The compiler also
keeps qualification eligibility false. Thirteen focused tests, Python
compilation, and diff checks were reported passing by the implementer.

## Review and limits

Independent review approved exact commit `39976d4aa64816facec21988e12da29bb07dcea5`.
The reviewer checked the corrected checkpoint/config association, all-five
track handling, protocol consistency, seed and coverage guards, and explicit
false qualification fields. This is an offline input compiler only: no
simulator, training, dashboard, held-out result, or qualification evidence
was produced. Only trusted local MECO schema-1 checkpoints are supported.
