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

Collection, fitting, and training require an explicit available indexed GPU
device (`cuda:N`, including ROCm/HIP devices exposed through PyTorch). Simulator
and model compute runs on that device; orchestration, tensor serialization,
checkpoint map-location, and RNG handling may use the host CPU.

Example using seeds assigned to the protocol's training partition:

```sh
python -m frc_defense.meco_supervised collect \
  --output evaluations/meco-supervised/train.jsonl \
  --scenario pickup_simple --category single --seeds 101 102 \
  --qualification-protocol qualifications/meco/protocol.json --device cuda:0
python -m frc_defense.meco_supervised warm-start \
  --track C1 --dataset evaluations/meco-supervised/train.jsonl \
  --output checkpoints/meco-warmstart/C1.pt \
  --qualification-protocol qualifications/meco/protocol.json --device cuda:0
python -m frc_defense.meco_track_training train --track C1 \
  --protocol-id <training-protocol-id> \
  --qualification-protocol qualifications/meco/protocol.json \
  --warm-start checkpoints/meco-warmstart/C1.pt \
  --output checkpoints/meco-five/C1 --device cuda:0
```

For a new five-lane campaign, pass the frozen qualification artifact and all
five track artifacts to `frc_defense.meco_supervisor`:

```sh
python -m frc_defense.meco_supervisor --seed 20261024 \
  --devices cuda:0,cuda:1 --evaluation-device cuda:0 \
  --qualification-protocol qualifications/meco/protocol.json \
  --warm-start-artifacts '{"A1":"checkpoints/meco-warmstart/A1.pt","A2":"checkpoints/meco-warmstart/A2.pt","A3":"checkpoints/meco-warmstart/A3.pt","C1":"checkpoints/meco-warmstart/C1.pt","C2":"checkpoints/meco-warmstart/C2.pt"}'
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
  --qualification-protocol qualifications/meco/protocol.json --device cuda:0
```

This is a label collector, not an expert policy or simulator oracle. The v1
`warm-start` command still consumes only v1 observed-behavior rows; v2-to-model
fitting is a distinct command and artifact path:

```sh
python -m frc_defense.meco_supervised warm-start-counterfactual-v2 \
  --track C1 --dataset evaluations/meco-supervised/counterfactual-v2.jsonl \
  --output checkpoints/meco-warmstart/C1-counterfactual-v2.pt \
  --qualification-protocol qualifications/meco/protocol.json \
  --device cuda:0
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
  --device cuda:0
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
ticks, configured P mode, and an applied flag plus SHA256/shape/dtype/scale for
the effective frozen map after executor dtype conversion. A changed map,
scale, goal, action, or tensor version forces an immediate focal-only replan
at the next macro boundary; later ticks
in the same macro do not repeat it. The signature includes map tensor identity,
version, scale, goal, and action. P0 records null map identity and scale. The
map hash identifies the map used for the latest focal route; if action masking
prevents applying a new map, the prior route's map is reported with its applied
goal/action signature, until an unweighted focal replan clears that association.
This is the latest route snapshot after a macro, not every within-macro replan
and not an exact actuator trajectory.

This is code-path integration only. No simulator replay has exercised or
inspected a P-shaped route from this seam yet, and the independent
`experimental_route_evaluation.evaluate_routes` probe still returns path
geometry without executing it through physics. The integration therefore does
not establish replay-backed P-route behavior or provide an oracle label.

The earlier U+B-only artifacts and validators continue to require a null P
target, zero P prior, and `p_mode=p0`. The later independent P-route oracle v3
schema is documented below; it binds sampled executed paths and is not
exhaustive legal-grid coverage.

#### Plan A legal-node tranche v1

The new `meco-counterfactual-plan-a-legal-node-v1` collector and
`meco-plan-a-legal-node-u-b-weights-v1` fitter are separate from Plan A v3.
They bind fresh source-v2 dataset and manifest hashes, protocol and train
split, source identity, device, full support digest, canonical node order,
reachable mask, and full ordered pair feasibility layout. Every label comes
from strict prefix replay followed by a legal simulator action and the same
deterministic teacher continuation, horizon, and per-macro discount as the
source manifest. Planner-internal AD* costs are never targets.

Coverage is a sampled tranche. For each branch, `N` is the count of feasible
reachable legal nodes. The selector chooses at most `floor(N/16)` nodes,
capped by `floor(512 / (2 * feasible_behavior_count))`, in deterministic
heading-bin × graph-region strata; it includes the root when the root has a
valid behavior pair. For about 7,528 nodes and five feasible behaviors this
selects at most 51 nodes (about 0.68% of reachable nodes), not an exhaustive
pass. One descriptor per selected node/behavior is selected by uniform hash
rank from that behavior's `D` feasible descriptors. It records `1/D`
inclusion probability and inverse-probability weight `D` as provenance; U and
B are defined as expectations over uniform descriptor draws, so the fitting
loss does not multiply sampled returns by `D`. Two paired repeat
outcomes use the same repeat seed across alternatives, retain each return and
report paired-difference variance. Each branch record includes the repeat seed,
requested action, executed physics-tick action trace, branch tick count, and
terminal/timeout state. Preempted, remasked, or partial requested actions are
recorded as unavailable and excluded from U and B targets. Behavior pairs are
comparable only when both paired repeats have the same nonzero sign; exact
all-repeat zeros are ties and all other cases are uncertain. There is no
calibrated noise margin, so this sign agreement rule is the declared minimum.
The per-prefix hard limit is 512 simulator
branches; the deterministic dataset preflight bound is
`source_record_count × 512`, capped at 12,000. A larger bound is rejected
before collection and requires partitioning the source dataset.

At each selected node, one uniformly sampled descriptor return estimates each
feasible behavior's descriptor-average return, then feasible behavior means
receive equal weight for the U target.
U is withheld when any feasible behavior outcome is censored or unavailable;
unavailable and censored behaviors are reported separately. Independently
valid behavior pairs can still contribute B preferences. Exact ties are
retained with no preferred behavior. The fitter uses
`smooth_l1(U(node), U_target)` plus a
feasible within-node pairwise softplus loss on `U+B` logits; the shared U term
cancels in each B contrast. It applies exact per-stratum node inclusion
weights; descriptor inverse-probability weights remain provenance for this
uniform-draw estimand. P targets remain null with zero prior
until the independent P-v3 oracle supplies them. Fresh deterministic
recollection must byte-match the dataset before fitting; the dedicated
weights-only consumer rejects CPU and nonmatching track/device/P contracts.

Planner and implementation are complete; scoped static review approved commit
`f91d76237`. Static compilation, diff and JSON checks only; runtime collection,
fitting and simulator evidence remain pending. No tests or simulator runs were
performed.

#### Joint Plan A U+B+P warm start v1

`warm-start-plan-a-joint-ubp-v1` trains one new A1/A2/A3 model and shared
encoder against legal-node U regression, centered feasible-behavior B
preferences, and independent oracle-v3 executed-route P preferences in the
same optimizer step. The U loss is inverse node inclusion weighted Smooth L1;
B is a per-node mean of stable comparable pairwise softplus terms with the
same node inclusion weights; P is pairwise softplus over replayed base bump
cost plus learned scaled P path integral, with mean-P regularization. The
three task weights are each 1.0 and are recorded in the artifact. Existing
U+B-only artifacts retain `p_mode=p0` and null P semantics; P-only artifacts
retain their own schema and loading contract.

Each configured update is a full epoch: every legal-node and P dataset row is
visited at least once, and the shorter dataset cycles in fixed order. Per-task
loss summaries omit batches without that task's target mask.

The joint fitter requires the legal-node and P-route manifests to bind the
same exact trajectory-v3 source data bytes and manifest bytes. It also checks
protocol, source identity, train split, prediction mode/contract and indexed
device, and strictly replays both datasets before fitting. The new
weights-only artifact binds both dataset and manifest digests and loads only
on its indexed CUDA device. Training accepts it only when `p_mode=learned` and
`p_cost_scale` exactly matches the frozen P-route scale. Implementation is complete and scoped review approved integrated candidate
`05338ce72`; runtime evidence is pending. This pass ran static compilation,
diff and JSON checks only; no tests, collection, fitting, GPU training or
simulator evaluation were run.

#### Explicit Plan A goal source v1

`collect-plan-a-expert-goal-source-v1` writes a separate source artifact from
the immutable trajectory-v3 rows. At each exact replayed Plan A support it
selects the feasible pair with minimum finite `route_cost`, breaking ties by
canonical support pair index, then executes that action with the selected
legal node pose as an explicit planner goal. Replay rebuilds support and
reruns the rule before accepting the descriptor hash; action remasking or
native planner preemption fails collection or annotation loading. The source
binds each record to its complete v3 base row, state, support digest, selected
node/behavior/descriptor, and physics-tick action trace. The expert annotation
loader accepts this source path and requires the annotation manifest to also
bind `expert_goal_source_sha256` to its bytes.

This deterministic nearest-route rule creates a demonstrated feasible B
preference. It does not claim global optimality and provides no U target;
`u_status` is unavailable until an independent expert value oracle and
uncensored feasible coverage exist. It does not convert or modify v3.
Collection and replay require runtime simulator validation before the data can
be treated as empirically established.

`warm-start-plan-a-expert-goals-v1` is the explicit annotation intake and
weights-only fitter for A1/A2/A3. Its `--dataset` is a JSONL annotation file
with the `meco-plan-a-expert-goal-annotations-v1` manifest. Each row identifies
the exact scenario/category/seed/decision, snapshot and state hashes, legal
support digest, replayed goal descriptor, executed action and selected
behavior, selected canonical legal goal, censoring inventory, and an explicit
`u_status`. An available U must be finite and uncensored across all feasible
behavior support; unavailable or censored U remains masked. B is learned only
as a behavior preference among available behaviors at the annotated goal.
Every row must bind the exact trajectory-v3 replay, explicit-goal source,
legal-node dataset, train split, frozen protocol and annotation-authority file
hash. The loader rejects action-only or mismatched labels.

Example after collecting the explicit-goal source and preparing the strict
legal-node dataset and annotation manifest:

```sh
python -m frc_defense.meco_supervised collect-plan-a-expert-goal-source-v1 \
  --dataset evaluations/meco-supervised/counterfactual-v3.jsonl \
  --output evaluations/meco-supervised/expert-goal-source.jsonl \
  --qualification-protocol qualifications/meco/protocol.json --device cuda:0
python -m frc_defense.meco_supervised collect-plan-a-legal-nodes-v1 \
  --dataset evaluations/meco-supervised/counterfactual-v3.jsonl \
  --expert-goal-source evaluations/meco-supervised/expert-goal-source.jsonl \
  --output evaluations/meco-supervised/legal-nodes.jsonl \
  --qualification-protocol qualifications/meco/protocol.json --device cuda:0
python -m frc_defense.meco_supervised warm-start-plan-a-expert-goals-v1 \
  --track A1 --dataset evaluations/meco-supervised/expert-goal-annotations.jsonl \
  --legal-node-dataset evaluations/meco-supervised/legal-nodes.jsonl \
  --source-dataset evaluations/meco-supervised/expert-goal-source.jsonl \
  --annotation-source evaluations/meco-supervised/annotation-authority.json \
  --output checkpoints/meco-warmstart/A1-expert-goals.pt \
  --qualification-protocol qualifications/meco/protocol.json \
  --device cuda:0
```

The fitter replays the exact legal support before training, emits model weights
only, binds the annotation/source/legal data and manifest hashes, and validates
the artifact before returning. It initializes and verifies the P head at p0;
it does not turn the route-cost selection rule into U, P, or a global
optimality claim. No expert-goal data or weights have been collected or fitted
by this implementation change.
