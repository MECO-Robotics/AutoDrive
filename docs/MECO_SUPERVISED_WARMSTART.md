# MECO supervised warm start

`frc_defense.meco_supervised` defines `meco-supervised-trajectory-v1`, target
schema `meco-observed-action-target-v1`, and weights artifact schema
`meco-warm-start-weights-v1`. Collection writes a JSONL record per macro
decision plus a content-hashed manifest. Every row binds the exact source
identity and frozen protocol ID, scenario/category, train seed, sensor raster,
global inputs, candidate table, route/risk costs, deterministic teacher action,
and label provenance. Collection rejects unbound protocol IDs and non-train
data is never implicitly selected.

The bundled `teacher_actions` policy is a diagnostic sensed-state heuristic,
not an expert or simulator oracle. Its action label is used as follows:

- A1/A2/A3 train only the behavior head for observed actions 0–6. Action 7 is
  excluded from their behavior labels because this teacher uses it as a no-op
  hold and supplies no explicit Plan A drive goal. They receive no legal-node
  goal-value or traversal-penalty target from this dataset.
- C1/C2 increase the probability mass over feasible candidate rows sharing the
  observed action. They do not receive candidate gross-reward, pairwise, or
  counterfactual preference labels.

The collection and training source identities must match exactly, and both
must name the same frozen protocol. Warm-start artifacts contain model weights
and provenance only: no optimizer state, RNG state, update cursor, or replay
state. `train --warm-start` and exact `train --resume` are mutually exclusive.
Subsequent resume checkpoints retain the original initialization provenance.
For C1/C2, model input width includes both candidate values and their explicit
missingness indicators.

Example using seeds assigned to the protocol's training partition:

```sh
python -m frc_defense.meco_supervised collect \
  --output evaluations/meco-supervised/train.jsonl \
  --scenario pickup_simple --category single --seeds 101 102 \
  --qualification-protocol qualifications/meco/protocol.json
python -m frc_defense.meco_supervised warm-start \
  --track C1 --dataset evaluations/meco-supervised/train.jsonl \
  --output checkpoints/meco-warmstart/C1.pt \
  --qualification-protocol qualifications/meco/protocol.json
python -m frc_defense.meco_track_training train --track C1 \
  --protocol-id <training-protocol-id> \
  --qualification-protocol qualifications/meco/protocol.json \
  --warm-start checkpoints/meco-warmstart/C1.pt \
  --output checkpoints/meco-five/C1
```

For a new five-lane campaign, pass the frozen qualification artifact and all
five track artifacts to `frc_defense.meco_supervisor`:

```sh
python -m frc_defense.meco_supervisor --seed 20261024 \
  --qualification-protocol qualifications/meco/protocol.json \
  --warm-start A1=checkpoints/meco-warmstart/A1.pt \
  --warm-start A2=checkpoints/meco-warmstart/A2.pt \
  --warm-start A3=checkpoints/meco-warmstart/A3.pt \
  --warm-start C1=checkpoints/meco-warmstart/C1.pt \
  --warm-start C2=checkpoints/meco-warmstart/C2.pt
```

The supervisor's `protocol_id` identifies its complete training configuration;
`qualification_protocol_id` separately identifies the frozen split artifact.
Both identities and every warm-start file hash appear in supervisor status and
configuration hashes. The supervisor rejects partial track maps, mismatched
source/protocol artifacts, non-train scheduled seeds, or a changed artifact.
Each worker receives only its own track's weights artifact on the first chunk;
later chunks use exact resume and retain the original initialization metadata.

## Matched-prefix v2 counterfactual collection

`collect-counterfactual-v2` writes the distinct
`meco-counterfactual-trajectory-v2` / `meco-matched-prefix-gross-return-v2`
schemas. It never reads v1 rows as return labels. For each frozen train seed,
the collector records the deterministic teacher prefix, predecision simulator
tensor-state and observation digests, candidate and Plan A support digests,
selected candidate descriptors, and executed action traces. Each branch starts
from a fresh environment, replays that prefix, and aborts on any digest or
executed-trace mismatch. It uses the archived candidate descriptor without
reselecting after replay.

Plan C records gross macro and fixed-horizon simulator returns, discounted
horizon returns, and pairwise return preferences over a deterministic bounded
sample of valid candidate descriptors. Plan A records a deterministic bounded
sample of valid node × behavior × descriptor pairs with gross returns and
pairwise preferences. Its P target is explicitly null with zero prior because
the current simulator executor does not execute P-modified routes. The common
continuation policy is the deterministic sensed-state teacher, and the artifact
binds its source identity, runtime/device, frozen protocol/train split,
candidate boundary, Plan A support order, horizon, and discount.
The discount is explicitly per macro decision (`gamma_per_macro`); the branch
reward has exponent zero and each continuation decision increments it once.
A final decision may use fewer than `--action-ticks` physics steps to finish
the fixed physics-tick horizon, and it still receives one macro discount.
The branch transcript and prefix digest contain only decisions before the
branch; the current decision's state and descriptor are stored separately.
Parity hashing covers the environment and simulator object state, reward
metrics, planner, fuel-physics object, and Python/NumPy/PyTorch RNG states. An
unknown state object fails collection instead of being silently omitted.
Mappings and sets are traversed in stable immutable-key order before reference
IDs are assigned. Object state includes instance dictionaries and inherited
slots; objects with opaque native bases fail closed. The optional HIP fuel
wrapper's Python buffers are included, and its native extension is bound by a
binary hash after inspection confirmed it exposes stateless kernel functions.

Example (start with a small train seed set; each decision branches multiple
simulator rollouts):

```sh
python -m frc_defense.meco_supervised collect-counterfactual-v2 \
  --output evaluations/meco-supervised/counterfactual-v2.jsonl \
  --scenario pickup_simple --category single --seeds 101 \
  --qualification-protocol qualifications/meco/protocol.json
```

This is a label collector, not an expert policy or simulator oracle. The v1
`warm-start` command still consumes only v1 observed-behavior rows; v2-to-model
fitting is a distinct command and artifact path:

```sh
python -m frc_defense.meco_supervised warm-start-counterfactual-v2 \
  --track C1 --dataset evaluations/meco-supervised/counterfactual-v2.jsonl \
  --output checkpoints/meco-warmstart/C1-counterfactual-v2.pt \
  --qualification-protocol qualifications/meco/protocol.json \
  --device cpu
```

The v2 fitter verifies dataset bytes, manifest and row schemas, protocol/train
split, source/runtime/device, candidate and support contracts, finite returns,
and pair labels. It replays every matched teacher prefix and rejects any
simulator, observation, candidate, support, descriptor, or action-trace parity
mismatch. It trains only on non-tied within-prefix pairwise comparisons of
`gross_horizon_return`. Plan A scores exact sampled support rows with U(node) +
B(behavior); P remains null and has zero prior. Plan C scores the exact replayed
candidate rows with the existing candidate scorer, but applies pairwise loss
to raw candidate utility because the labels are gross returns; route and risk
costs are excluded from this target loss. The resulting
`meco-counterfactual-warm-start-weights-v2` artifact is weights-only and has a
separate `validate_counterfactual_warm_start_artifact` /
`load_counterfactual_warm_start` API. The v1 `warm-start` and its artifact
contract remain unchanged and never read v2 labels. Plan A's penalty head gets
no v2 pairwise gradient and remains untouched in the artifact; `plan_a_p_mode`
is explicitly `p0` for this warm-start inference contract. Later RL can train
the preserved head normally, and P ablations remain independent.

No simulator run has been performed to establish that reset-and-replay parity
succeeds on the configured runtime/device. A successful v2 fit requires that
replay to pass when the command is run; Phase 2 still needs a representative
end-to-end simulator evaluation before its behavior claims are treated as
validated.

### Plan A direct value warm start (v3 artifact)

Plan A can use the same strict, replay-bound v2 counterfactual dataset with a
direct regression objective:

```sh
python -m frc_defense.meco_supervised warm-start-plan-a-v3 \
  --track A1 --dataset evaluations/meco-supervised/counterfactual-v2.jsonl \
  --output checkpoints/meco-warmstart/A1-plan-a-v3.pt \
  --qualification-protocol qualifications/meco/protocol.json \
  --device cpu
```

Collect that v2 dataset from this exact source checkout first; its source
identity is checked during fitting, so datasets collected before the v3 code
change are rejected rather than silently reused.

This command accepts A1/A2/A3 only. On each replayed branch it fits
`U(node) + B(behavior)` to the descriptor's archived `gross_horizon_return`
with smooth-L1 loss and retains the within-prefix pairwise preference loss.
Only the deterministic sample of feasible node×behavior descriptors archived
from the full, ordered Plan A support is directly value-supervised; the v3
artifact binds the v2 dataset and manifest digests, support-order contract,
source identity, frozen protocol/train split, prediction mode, runtime/device,
horizon, and discount semantics. This is sampled support coverage, not a claim
that every legal node was labeled.

The v3 artifact is weights-only and is consumable through the existing generic
warm-start validator/loader. Its P target remains null because the current
counterfactual executor does not execute P-modified AD* routes. The fitter
zeros the unlabelled P head and records `p_mode=p0`; later RL can train P. A
replay executor that executes and records P-shaped routes plus an independent
oracle route-cost target is still required to close Plan A traversal-penalty
supervision in Phase 2. No P target is inferred from U/B returns.

#### Route-execution and coverage gate

The MECO Plan A call path now passes its selected projected P map and scale
through the macro executor into `TensorThreeVsThreeEnv.step`. On each native
replan, the same frozen map is applied to the focal neural robot's route; other
robots retain their unweighted AD* calls. A failed depot route is retried with
the same P map when the focal route is eligible. P0 passes no map and retains
the existing route-planning calls. The route override validates its map shape,
finite nonnegative values and scale once per unchanged frozen map, neural
slot-3 ownership, and paired goal action. Standard Plan A train/eval invokes
the P-map path eagerly. If a caller supplies a P map while a CUDA stream is
being captured, `step` raises `ValueError` and aborts that call; it does not
fall back silently. The dashboard's existing captured route path supplies no
P map and remains unchanged. No fallback or graph-compatible P path has been
validated.

Evaluation records now have a distinct `executed_planner_route_trace` for
A1/A2/A3 and their P0 runs. Each macro entry snapshots focal slot-3
`last_path[:last_lengths]` and the observed planner goal after macro execution,
with selected goal, selected action, actual per-physics-step action trace,
ticks, P mode, and SHA256/shape/dtype/scale for the map actually passed to
the macro. P0 records null map identity and scale. This is the latest route snapshot after a
macro, not every within-macro replan and not an exact actuator trajectory.

This is code-path integration only. No simulator replay has exercised or
inspected a P-shaped route from this seam yet, and the independent
`experimental_route_evaluation.evaluate_routes` probe still returns path
geometry without executing it through physics. The integration therefore does
not establish replay-backed P-route behavior or provide an oracle label.

For that reason this checkout still does not define or accept a P-route-oracle
or exhaustive-route-coverage schema. The v2/v3 artifact validators continue to
require a null P target, zero P prior, and `p_mode=p0`; a sampled subset of
legal Plan A support descriptors remains sampled coverage, not exhaustive
node/path coverage. A versioned oracle contract requires a representative
simulator replay that proves the selected map affects only the focal native
route, records the executed route and map provenance, and matches the replayed
route costs to an independent reference. Until then, P labels and exhaustive
P-route coverage remain unavailable and must not be inferred from planner API
capability or U+B return labels.
