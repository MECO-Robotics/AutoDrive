# MECO track adapter status

`frc_defense/meco_tracks.py` contains isolated experimental model and scoring
interfaces for the five tracks in the [evaluation plan](MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md).
These modules do not change the production controller or make any promotion or
deployment claim. Paired evaluation includes the production deterministic D0
slot-3 controller at native 20 ms cadence; it is not constrained by the MECO
candidate table.

## Implemented in this slice

- **A1:** multiscale raster U-Net + GRU, producing eight heading-bin utility
  fields, nonnegative traversal penalties, and five behavior logits.
- **A2:** structured SE(2) graph U-Net with fine and coarse directed transition
  message passing, connectivity-aware region pooling, regional self-attention,
  GRU context, skip fusion, and per-node utility/penalty outputs. Heading bins
  use circular sine/cosine encoding; 7-to-0 wraparound is continuous.
- **A3:** hierarchical SE(2) graph message passing, 64-region pooling and
  attention, GRU context, and per-node utility/penalty outputs.
- **C1:** shared candidate-feature scorer with seven-scalar global encoder and
  GRU context; returns candidate utility and an uncertainty estimate.
- **C2:** shallow raster encoder, candidate-to-map cross-attention, GRU
  context, and candidate utility/uncertainty outputs.
- A common adapter converts model scores to a **masked categorical
  distribution**, so stochastic callers can obtain action, log probability,
  and entropy for PPO. Missing feature values are accompanied by explicit
  unavailable indicators.
- A stale/mismatched-output gate checks a unique UUID episode generation,
  exact observation and candidate content digests, candidate ordering,
  timestamp age, and model identity. Bound scorer outputs use schema v2 with
  required SHA-256 digests. Invalid output uses the listed
  valid deterministic fallback. Scores cannot change the candidate mask.
- `build_safe_action_candidates` exposes the curriculum simulator's existing
  eight masked action rows in canonical row order. `deterministic_candidate_scores`
  and `predict_opponent_endpoint` provide a small matched D1/D2 comparator seam
  over those same rows.
- **Plan A PPO support:** A1/A2/A3 sample from a fixed node-major tensor over
  every legal graph node crossed with drive, four target-bound intake choices,
  score, ferry, and defense descriptors. A model-independent shortest-path
  pass fixes reachable nodes, physical distance, heading-weighted route cost,
  and a parent tree including the root-pose connector. The support mask/order
  and selected pair descriptor are
  retained for PPO log-probability recomputation. `candidate_k` applies only to
  Plan C.

## Deliberate limits

The Plan C/D1/D2 candidate service keeps the original stable K=8/16/32 prefix
over eight action primitives and six collection offsets. C1/C2 may append one
`continue` row after that prefix when a task commitment is active; the row
recomputes target matching, phase legality, endpoint safety, and AD* route cost
from the current snapshot. Commitments expire after ten macro actions and are
cleared on target loss, invalid route/action, goal completion, episode reset,
or expiry. Intake commitments match the same sensed fuel by position rather
than relying on the changing top-four buffer index. D1/D2 retain their prior
per-macro behavior without commitment carryover.

The selectable table is bound to its candidate IDs, features, masks, route
costs, and commitment descriptor by a SHA-256 digest retained in PPO rollout
rows; C1/C2 log probabilities are recomputed against that frozen table before
updates. K8 ordinary row IDs/order remain unchanged. Candidate coverage is
measured against an independently enumerated, route-qualified expanded goal
set (32 stable rows plus 18 additional offsets per sensed target). It reports
feasible-row coverage and selected/oracle minimum and mean route costs, not
regret or simulator optimality.
Static endpoint clearance encloses the robot and extended intake at every yaw.
A sensed-target bearing supplies a nonbinding `capture_estimate`; it is not a
feasibility mask because the native controller does not command or guarantee
the inferred terminal heading. Dynamic future swept-robot collision remains
outside this candidate feasibility check, and the simulator's native
intake/action executor remains authoritative.

Plan A and Plan C share the deterministic legal SE(2) graph inputs for A2/A3
(`node_features`, directed `edges`, `edge_mask`, and `region_ids`). The bounded
candidate adapter's sensed-target heading behavior applies to Plan C and the
D1/D2 controls; it does not define A1/A2/A3's stochastic action support.

For A1-A3, the nonnegative Plan A traversal field is projected to the XY
planner grid by averaging only legal heading bins at each cell. The stochastic
policy combines `U(node)`, the selected behavior logit, deterministic ETA/risk
costs, and the current model's P integral over the frozen legal SE(2)
shortest-path tree. Binary-lifting parent tables let PPO recompute all node
integrals with O(N log N) tensor gathers rather than per-candidate planner
calls. `p_mode=p0` remains a zero-P matched ablation. This tree is a static
legal-graph reachability/cost reference; it is not the simulator's execution
route. The simulator still uses XY AD* with swept SE(2) checks and dynamic
robot arbitration. The MECO Plan A training and evaluation call path now passes
the projected P map through each macro to native AD*, using the same frozen
map on focal slot-3 replans and eligible depot retries; other robot routes
remain unweighted. P0 supplies no map and preserves the existing planner path.
This wiring has not yet been exercised or inspected in a simulator replay, so
it is not runtime evidence for P-shaped native routes. The isolated weighted-
route probe still returns geometry without executing it in physics. No
P-route oracle or exhaustive route-coverage labels are defined; v2/v3
supervised artifacts continue to fail closed with P null, zero prior, and
`p_mode=p0`, and their sampled support is not exhaustive node/path coverage.
The standard Plan A train/eval call path uses eager steps. A P map supplied
during CUDA stream capture causes `TensorThreeVsThreeEnv.step` to raise
`ValueError` and abort that call; it does not silently fall back. Dashboard
capture currently supplies no P map. The environment and AD* planner cache
finite/nonnegative and scaled-finiteness checks per unchanged map tensor
identity/version/scale, avoiding repeated scalar synchronization across macro
replans. Shape and scale checks still run per call. This capture behavior has
not been runtime validated.
Intake support binds a sensed target slot and a
bearing-aligned capture envelope; score support is restricted to clear score
grid endpoints and heading bins within 5 degrees of the HUB-facing direction;
ferry support requires carried fuel, a controllable current ferry endpoint,
and a heading inside the simulator's 45-degree release cone; carrying dumper
ferry pairs are excluded because native dumper action 6 can release fuel
anywhere before reaching the selected endpoint. Evaluation records any observed
early slot-3 release before the selected endpoint as an execution event.
Defense support is restricted to the
current intercept/post endpoint. Each selected graph heading is passed as an
explicit terminal target. At the XY endpoint, the controller pauses
translation, samples static rectangular rotation clearance every 5 degrees,
and checks observed robots with circumscribed footprints, stopping drift, and a
constant-velocity point forecast over estimated turn time. The forecast has no
uncertainty inflation, and the 5-degree samples do not prove continuous swept
clearance; native route and collision checks remain authoritative. During PPO
training, every macro action is the action sampled at that observation and its
own sampled log probability; the next macro boundary samples a fresh action.
This keeps the executed action paired with its policy density and reward. An
incomplete goal at a macro or rollout boundary is not itself an execution
failure. A sampled deterministic fallback, native action remask, or native goal
preemption remains critic-only and is counted by reason in update stats; none
is credited as a successful execution. C1/C2/D1/D2 retain their existing
per-macro action semantics. Evaluation may retain a selected Plan A goal over
macros, as recorded in the evaluation contract. Intake headings must satisfy
their target capture geometry. P integrates the root-pose-to-root-node
connector as well as graph edges.
Action 7 remains a native no-op hold. The Plan A adapter permits it to traverse
only when paired with the selected frozen planner goal; the existing route,
obstacle, and action-safety checks remain in force. Dumper drive pairs are
masked while carrying fuel because native action 7 redirects that role to ferry.
If the native action executor remasks the sampled action during a macro, the
transition is recorded as an execution failure and excluded from the PPO actor
loss while remaining available to the critic. Dynamic route mismatch beyond
action remasking is not yet separately qualified. C1/C2 keep their bounded
candidate-only scoring path.

The code now has a matched PPO/evaluation harness and a resumable five-lane
supervisor, but no training/evaluation run is claimed here. Its task scheduler
samples the registered curriculum DAG through `task_mixture`, persists a
shared task/category/seed assignment across all five learned tracks, tracks
category exposure, and biases practice toward recent failed categories plus
their ancestors. Stage unlocks mean exposure only; they do not update
`CurriculumState.qualified` or satisfy outcome gates. The periodic supervisor
evaluation remains a diagnostic comparison and does not implement the full
held-out defender-family protocol. Baseline freezes, qualified retention,
full-game metrics, simulator records, confidence-interval promotion gates,
full Phase 2 expert goal-value supervision, Hailo/ONNX export, and hardware
profiling remain open. The separate v2 matched-prefix gross-return pairwise
fitter and weights-only loader are implemented alongside the observation-only
v1 path. They are not behavior-validated: no v2 dataset has been replayed and
fit here, and no representative end-to-end simulator evaluation has been
reviewed. See [MECO_SUPERVISED_WARMSTART.md](MECO_SUPERVISED_WARMSTART.md) for
the v1/v2 commands and contracts.

## Adapter calls

```python
model = make_meco_track("C1", candidate_features=len(table.feature_names))
result = score_candidates("C1", model, raster=raster, globals_=globals_,
                          table=table, hidden=hidden)
action, log_probability, entropy = sample_candidate_action(result["logits"])
```

For A2, `score_candidates` requires fine and coarse graph transition tensors;
A3 requires fine graph tensors. Node coordinates must be normalized to the
field, heading bins must be integer values from 0 through 7, and valid pose and
node indices are rejected if they fall outside their declared domains. Empty
regions are excluded from attention. Preserve the returned GRU `hidden` state between decisions and
reset it on episode reset or tracking-loss events. Resetting is the caller's
responsibility in this prototype.

No simulator behavior evaluation or dashboard replay is claimed by the
focused model/contract tests. Any game-behavior training or evaluation must
retain and inspect an end-to-end simulator record under the repository's
`AGENTS.md` validation rules.

## Section 6 collection-action bouts

The typed episode metric ledger schema v3 records contiguous focal slot-3
executed actions 0 through 3 as collection-action bouts. Each closed bout keeps
its zero-based start and exclusive end outer-step ticks, executed action
sequence and per-action counts, focal acquired quantity, acquisition result,
and closure reason. A bout is acquired only when the focal
`fuel_acquired_event` quantity increases during that bout. Non-collection
actions close a bout; episode finalization, timeout, or evaluation-error
cleanup closes any active bout. The record includes bout count, acquired
count, and acquired proportion; the proportion is null when no bouts occurred.
Missing executed-action or acquisition signals make this metric unavailable
instead of reporting zero. HUB denied remains a distinct simulator denied
event, while abandoned fuel remains null because this evaluator has no
authoritative abandoned-event producer. This implementation has only received
static checks; simulator-record validation remains outstanding.
