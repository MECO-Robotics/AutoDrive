# Five-track candidate feature schema fixture

The approved plan is `docs/plans/five-track-candidate-feature-schema-20261025.md`
(planner head `6d69275fa0c187574d0293bef2228daba1ead88c`). It repairs stale test
fixtures without changing the registered 20-column C1/C2 runtime schema.

Implementation candidate and integrated mainline commit:
`2fb402c69475ab3f43707a5eb0ec797e52b5fa8c`. The change is limited to
`tests/test_meco_tracks.py`: fixtures now derive ordered fields from
`SAFE_ACTION_FEATURES + EXTRA_FEATURES`, populate declared values, and provide
valid graph node indices for A2/A3. Negative cases retain fail-closed checks for
wrong names, wrong order, and malformed row width.

Independent review approved exact head `2fb402c69475ab3f43707a5eb0ec797e52b5fa8c`.
The implementer reported 17 focused tests passing and `git diff --check` clean;
the reviewer inspected source but did not execute tests. This is synthetic
schema/model-shape coverage only. It does not establish simulated robot
behavior, scoring, runtime readiness, or qualification.
