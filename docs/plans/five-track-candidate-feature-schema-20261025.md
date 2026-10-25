# Five-track candidate feature fixture repair plan

**Status:** planner proposal only. No source or test code was changed, and no test, simulator, GPU, training, dashboard, or data action was run.
**Planning base:** `7e7e7253c9756ef17bc6b33582921e19cd645556`.
**Primary source plan:** `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`.
**Implementation-status source:** `docs/MECO_TRACKS_IMPLEMENTATION_STATUS.md`.
**Related contract:** `docs/implementation-provenance/meco-input-provenance-contract.md`.

## Goal

Repair the confirmed baseline failure in
`tests/test_meco_tracks.py::test_five_architecture_outputs_and_candidate_logits`
by making that five-track fixture use the exact canonical named candidate-row
schema already registered for C1/C2. Preserve the production validator,
feature registry, model inputs, all five architectures, and scoring behavior.
The change is a test-fixture repair; it does not establish simulator behavior or
complete the five-track study.

## Source audit and diagnosis

At the planning base, the test helper `table()` supplies only six raw feature
columns (`goal_x_m`, `goal_y_m`, `heading_bin`, `behavior_index`,
`field_length_m`, `field_width_m`). For A2/A3 it appends
`legal_node_index`, still leaving the fixture short of the registered Plan C
schema. `score_candidates()` intentionally rejects any C1/C2 table unless
its feature names exactly equal
`SAFE_ACTION_FEATURES + EXTRA_FEATURES` and every row has the registered
20 raw values. Therefore the test fails at C1 before candidate scoring; C2
would reject the same fixture. This is a stale abbreviated fixture, not a
runtime candidate-builder/schema defect.

The authoritative raw C feature order is the concatenation of
`SAFE_ACTION_FEATURES` and `EXTRA_FEATURES`: 20 named values, including
`legal_node_index`. The serialized C1/C2 tensor is those values followed by
20 missingness indicators (width 40), as checked by
`frc_defense/meco_input_provenance.py`. The candidate builder emits this raw
table and appends the six `EXTRA_FEATURES`; do not modify this order, width,
registry digest, provenance, or exact-name/row-width check.

Plan A has a different neural representation: A1 uses raster outputs and A2/A3
use legal-graph outputs, then gather candidate scores using named table columns.
They do not consume C1/C2's 40-value model tensor. The test can give all five
tracks the same canonical raw candidate table while preserving each track's
existing neural input and lookup path. For A2/A3, set the canonical
`legal_node_index` values to valid indices in the eight-node test graph;
do not append a duplicate column or reorder the table.

## Related in-flight branches and dependency boundaries

The in-flight predictor implementation branch at
`codex/phase0-multimodel-predictor-implementation-20261025` (observed head
`eb0807033f3bd75c1f21ed97f58dc5bc99965af`) adds the `multi_model_v1`
forecast contract and routes forecast risk through candidate construction.
Its diff changes prediction/risk semantics in `meco_tracks.py`,
`meco_candidates.py`, training, ablation, docs and predictor tests; it does
not change `SAFE_ACTION_FEATURES`, `EXTRA_FEATURES`, their order, or the
registered 20/40 widths. The fixture repair is independent of predictor
semantics. If implementation starts after that branch integrates, re-audit
the exact base and keep the fixture bound to the current registry rather than
copying candidate-branch files.

The paired-calibration implementation branch at
`codex/phase0-resumable-paired-calibration-20261025` (observed head
`77463fc882b24726d5cbd0086fe76199ca8479c1`) extends
`meco_qualification.py` and paired-calibration contract tests. It depends on
the predictor contract for executable `D2/multi_model_v1` identity, but it
does not read or change the unit fixture's feature columns. No calibration,
qualification, or paired campaign edit is needed for this repair.

## Narrow implementation steps

1. Rebase the implementation worktree on the then-current reviewed mainline
   and re-confirm the failing test's exact fixture/validator relationship.
2. Change only the local `tests/test_meco_tracks.py` fixture/helper needed to
   construct rows from the runtime raw schema
   `tuple(SAFE_ACTION_FEATURES) + tuple(EXTRA_FEATURES)`. Populate values by
   feature name, including the six current base-action fields and the
   route/commitment fields; use `None` for unavailable estimates so the
   candidate encoder exercises its declared missingness path.
3. Use the same 20 feature names and width for A1/A2/A3/C1/C2. Set valid
   `legal_node_index` values in graph-mode rows; do not append a second
   legal-node field. Keep the same two candidate IDs/mask and preserve existing
   field bounds, heading, behavior, graph tensors and output assertions.
4. Leave `score_candidates()` exact C1/C2 schema validation, the provenance
   registry, candidate-builder schema, predictor semantics, score costs,
   model architecture/inputs, and checkpoint contract untouched. If the
   implementer finds a real runtime schema mismatch, stop and request a
   separately scoped planner pass instead of relaxing the validator or
   silently widening the feature registry.
5. Commit and push this fixture-only checkpoint, then have an independent
   reviewer inspect the exact pushed head before integration.

## Acceptance evidence to gather in implementation/review

- Run
  `tests/test_meco_tracks.py::test_five_architecture_outputs_and_candidate_logits`
  and confirm A1/A2/A3/C1/C2 each return two masked logits and valid categorical
  action/log-probability/entropy shapes.
- Run the focused `tests/test_meco_input_provenance.py` checks to confirm the
  authoritative ordered 20 raw / 40 serialized feature contract and digest
  are unchanged.
- Add or retain a fail-closed assertion for a malformed C1/C2 schema (wrong
  name/order or row width) if the focused suite does not already cover it.
  This guards against fixing the fixture by weakening production validation.
- Confirm the diff changes test fixture construction only; no candidate-builder,
  model, predictor, training, checkpoint, qualification, or calibration code
  changes.
- Have the independent reviewer record the exact reviewed commit and verdict.

This fixture-only test exercises synthetic tensor/model scoring and does not
run a simulated game. If implementation expands into candidate construction,
robot action, FUEL scoring/collection, or any other simulator behavior, follow
`AGENTS.md`: run the changed path in a representative seeded end-to-end
simulator evaluation, inspect and retain its record/replay and scenario/seed,
and before replay review restart the dashboard services on the current
integrated checkout and verify active services, endpoint responses, current
backend/frontend assets, and fresh replay provenance. Report the dashboard
verification and replay ID, or explicitly report any gate that could not be
completed. Do not interrupt the live r4 run or contend for its WX9100 devices;
defer device-dependent gates until the user manually stops training and both
devices are actually released.

## Explicit non-goals

- No change to the 20 raw candidate values or their 40-wide C1/C2 serialized
  tensor, registry digest, model architecture, training/checkpoint identity,
  candidate ordering/masks, or inference contract.
- No cross-architecture claim that Plan A serializes or consumes Plan C's
  feature tensor.
- No implementation of `multi_model_v1`, no predictor/calibration changes,
  no simulator behavior change, and no edits to the live r4 campaign or data.
- No claim of simulator validation, learning benefit, qualification, or
  completion of the five-track evaluation plan.
