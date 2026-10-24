# Phase 2 Plan A human preference labels v2

## Label contract

The v2 intake is separate from the existing `warm-start-plan-a-expert-goals-v1`
path and leaves its schema and behavior unchanged. A v2 authored JSONL row has
the branch identity plus exactly these fields:

```json
{"scenario_id":"pickup_simple","category":"single","seed":103,"decision_index":0,
 "goal_preference_tiers":[[12,18],[24],[31,33]],
 "behavior_preferences":[{"node_id":12,"left_behavior_id":1,
   "right_behavior_id":2,"preferred_behavior_id":2}]}
```

Each tier lists tied preferred legal goal nodes; earlier tiers rank above later
tiers. The materializer expands cross-tier pairs for U ranking. Behavior labels
are explicit pairwise preferences among available behaviors at a specific
legal node. Labels are human-authored declarations only: the authority file's
`source_id` is recorded, but identity, expertise, and optimality are not
authenticated or claimed. A preference may select a node independently of the
deterministic teacher's executed choice, provided it is in exact replayed legal
support.

The strict parser rejects unknown/missing fields, duplicate identities or
preferences, unsupported nodes/behaviors, incomplete rankings, held-out seeds,
and source/support/protocol/split mismatch. Both the materializer and fitter
bind dataset and manifest digests. The fitter replays the legal-node dataset
and requires byte-equivalent regenerated support before it trains.

U is fitted only with pairwise ranking loss; no numeric U is supplied or
inferred. Numeric U remains unavailable unless a separate independent oracle
provides complete uncensored values. B uses within-node centered pairwise
behavior preferences. P remains at p0 and has no target. The A1/A2/A3 output is
a weights-only artifact: optimizer state and resume cursor are excluded.

## Workflow

First produce a replay-validated training counterfactual dataset and its
matching legal-node dataset using the existing collection commands. Prepare
authored labels with the exact row schema above, and a declared authority JSON
such as `{"source_id":"panel-2026-01"}`. Materialize the v2 labels:

```bash
python -m frc_defense.meco_plan_a_preferences_v2 \
  --authored-rows annotations.jsonl \
  --source-dataset counterfactual.jsonl \
  --legal-node-dataset legal-nodes.jsonl \
  --qualification-protocol protocol.json \
  --authority-json authority.json \
  --output plan-a-preferences-v2.jsonl
```

Fit one weights-only artifact for A1, A2, or A3 on the same indexed CUDA device
bound to the collected data:

```bash
python -m frc_defense.meco_supervised warm-start-plan-a-preferences-v2 \
  --track A1 --dataset plan-a-preferences-v2.jsonl \
  --legal-node-dataset legal-nodes.jsonl \
  --source-dataset counterfactual.jsonl \
  --authority-json authority.json \
  --qualification-protocol protocol.json \
  --device cuda:0 --output A1-preferences-v2.pt
```

Substitute `A2` or `A3` to fit the corresponding model. This command performs
fresh simulator replay while validating support; the contract test uses replay
stubs and does not run simulator scenarios.

## Validation and limits

Focused CPU contract tests cover deterministic materialization, pair expansion,
independent goal selection, unavailable numeric U, unsupported labels,
held-out rejection, and the replay-validation loader boundary. No simulator,
GPU training, or qualification run is part of this implementation. Labels do
not prove annotator expertise or goal optimality. The weights are a warm-start
artifact, not a completed policy qualification or deployment approval.
