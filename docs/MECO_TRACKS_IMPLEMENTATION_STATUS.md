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
over eight action primitives and six collection offsets. C1/C2/D1/D2 share one
candidate table and may append one `continue` row after that prefix when a task
commitment is active; the row
recomputes target matching, phase legality, endpoint safety, and AD* route cost
from the current snapshot. Commitments expire after ten macro actions and are
cleared on target loss, invalid route/action, goal completion, episode reset,
or expiry. Intake commitments match the same sensed fuel by position rather
than relying on the changing top-four buffer index. D1/D2 now carry the same
commitment lifecycle as C1/C2. All four subtract the same fixed 0.10-point
switch cost from feasible non-continue rows while a commitment is active; D1
and D2 differ only in hand utility and their selected prediction mode. Route
cost, prediction risk, candidate eligibility, expiry and hysteresis are frozen
in evaluation provenance. This parity is implementation evidence only; it has
not yet been simulator-validated.

The selectable table is bound to its candidate IDs, features, masks, route
costs, and commitment descriptor by a SHA-256 digest retained in PPO rollout
rows; C1/C2 log probabilities are recomputed against that frozen table before
updates. K8 ordinary row IDs/order remain unchanged. Candidate coverage is
measured against an independently enumerated, route-qualified expanded goal
set (32 stable rows plus 18 additional offsets per sensed target). It reports
feasible-row coverage and selected/oracle minimum and mean route costs, not
regret or simulator optimality.
Static endpoint clearance encloses the robot and extended intake at every yaw.
For collection candidates, a bearing-aligned capture envelope built from the
sensed target and candidate goal is also an eligibility filter. It uses the
same effective fuel radius as the native collector predicate
(`env._fuel_radius`) and fails closed when target geometry is unavailable.
The selected sensed track supplies the native pickup bearing throughout the
approach, and endpoint alignment is attempted only after native swept static
and observed-robot clearance checks. The candidate capture envelope remains
an eligibility estimate; target loss or a remasked paired action ends the
macro, and physical pickup remains simulator-authoritative. Dynamic future
swept-robot collision remains outside candidate feasibility checks.

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
legacy weighted-route probe still returns geometry without executing it in
physics. The
legacy v2/v3 supervised artifacts continue to fail closed with P null, zero
prior, and `p_mode=p0`. Separate Plan A P route-v2 supervision now supplies
sampled within-prefix route-preference targets from simulator returns; it
does not provide independent per-cell oracle labels or exhaustive route
coverage. Collection, fitting, and outcome validation remain pending.
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
is credited as a successful execution. C1/C2/D1/D2 use per-macro decisions and
shared candidate commitment carryover. Evaluation may retain a selected Plan A goal over
macros, as recorded in the evaluation contract. Intake headings must satisfy
their target capture geometry. P integrates the root-pose-to-root-node
connector as well as graph edges.
Action 7 holds unless the Plan A adapter pairs it with the selected frozen
planner goal; that explicit goal is the only way action 7 traverses, and the
existing route, obstacle, and action-safety checks remain in force. Legal
dumper traversal remains supported while carrying fuel. Carrying dumper ferry
action 6 remains excluded because it can release fuel before reaching the
selected endpoint.
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

## A1 empty-support failure follow-up

The observed A1 run `82d4f0f6fd3248e6b67e12c3a34ed132` failed at
`2026-10-23T14:15:01.060912+00:00` with `IndexError: too many indices for tensor
of dimension 1`. Its recorded source identity was
`fe53f457ccf295681c3fbd2bfb02b3958ef1cbce2596e427b28ce069bb9a4731` and its
protocol ID was
`05c6843d5355fb5a4a9d7e998b0381640338dafd5451227cc41cb51325ae642c`. The
failure occurred while assembling the empty-support diagnostic: the selected
ferry target is already an `[x, y]` pair, but diagnostic serialization indexed
it again as a batched tensor. The isolated fix serializes its two values
directly and requires exactly two finite coordinates. It does not change the
legal action mask, support tensor, sampling, or PPO behavior.

The builder now reaches and raises the typed `PlanAEmptySupportError`; callers
in counterfactual collection/replay do not convert it to an `IndexError` and
retain their existing environment cleanup. This source change requires a fresh
source identity and fresh output directory/checkpoints; the failed run cannot
be resumed under this code. No test or simulator evaluation was run for this
follow-up, so simulator validation remains pending.

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
regions are excluded from attention. Preserve the returned GRU `hidden` state between decisions.
Training and matched evaluation now reset it when an opponent track that was
available on the previous decision becomes unavailable or older than 2.0 seconds
according to raster channels 4/5 (opponent confidence/signed time). The transition
rule and threshold are recorded in run and evaluation protocol metadata. The same
post-reset hidden input is saved for PPO replay.

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

## Separate Plan A P route warm start (code path only)

The v2 counterfactual dataset and v3 Plan A U+B weights keep their explicit
P-null/P0 contracts. A separate `meco-counterfactual-plan-a-p-route-v2`
collector consumes a replay-bound v2 training dataset and executes a P0 branch
plus reproducibly generated, model-independent positive XY cost-map probes
through the native slot-3 AD* path. P0 and probes force the same focal selected-
goal refresh at the macro boundary; P0 applies no added traversal cost. Each
outcome binds to the exact selected descriptor hash. Its target is pairwise preference between
the actual routes, ranked by simulator gross horizon return under the same
prefix and deterministic continuation policy. It is not a per-cell oracle
label. Each branch records every distinct focal route snapshot within the
selected macro, plus the map-application state and action trace. The route
warm-start fitter replays every branch and route, trains only
the Plan A penalty output channel on differentiable integrated route costs,
and starts that channel at a small positive softplus value to avoid the v3
zero-head gradient dead zone. The resulting weights-only artifact has its own
schema and requires `p_mode=learned` when used to initialize training.

CLI entry points are `collect-plan-a-p-routes-v1` and
`warm-start-plan-a-p-v1`. This implementation has only received static checks;
no P-route dataset has been collected, no fitting has run, and no fresh
dashboard-backed simulator replay has verified the resulting behavior. The
current route coverage field counts sampled execution-path cells and explicitly
does not claim exhaustive legal-grid coverage. Fitting requires the sampled
paths to cover at least the configured fraction of legal XY cells (default
1%); the exact observed and minimum fractions are stored in the artifact.
P warm-start outcomes therefore
remain unqualified until the dataset, fit, and representative replay gates run.

## Runtime failure follow-up (2026-10-23)

The first live five-track batch stopped before a cycle completed. A2 failed at
42 updates after `pickup_simple/cluster` reached a state with a loaded dumper,
no currently sensed eligible FUEL target, and only native action 7 legal. Plan
A correctly failed closed because its drive support also excluded loaded
dumper traversal. The corrected support now retains action 7 at reachable
legal graph nodes when the native mask permits it; traversal still requires
the selected explicit planner goal and native route/safety checks, while
loaded-dumper ferry action 6 stays excluded. A1 stopped at 47 updates from
the supervisor's sibling-failure handling. Those checkpoints are source-bound
to the prior checkout and are not resumable under this code revision.

The first current-source counterfactual collector attempt also exposed that
`CurriculumEnv` did not expose the native perceived-track accessor used by
Plan C's expanded-target candidate builder. A shared native accessor now
returns only perceived track positions, velocities, and masks; both Plan C
target enumeration and commitment matching query slot 3, matching execution.
The failed collector run produced no validated dataset. The fixes have scoped
static review and compilation only; seeded simulator collection, route-v2
fitting, current-dashboard replay inspection, and a new all-five training run
remain pending.

The next current-source counterfactual collection exposed a digest failure on a
tensor with singleton dimensions and a non-unit final stride: `contiguous()`
preserved the degenerate stride, so the tensor could not be reinterpreted as
bytes. `_tensor_digest` now rejects non-strided layouts and clones supported
tensors into explicit contiguous format before hashing their flattened logical
bytes. The failed collection did not produce a validated dataset. This fix passed scoped review in isolated commit 6dd5a5311 and was integrated as
dc153a54a; collection and simulator validation remain outstanding.

The next collector attempt exposed a Plan C indexing error: the native
`_perceived_fuel(3)` API correctly returns batched `[world, track, ...]`
buffers, while the expanded-target and commitment consumers treated the first
dimension as a track dimension after partially indexing only the mask. Commit
`7ba2d89de` retains `env.n == 1`, validates aligned batched position, velocity,
and mask shapes, and explicitly selects world 0 for both consumers. The failed
attempt produced no validated dataset. This correction passed independent scoped review in isolated commits
7ba2d89de/f5bd75922 and was integrated as 2dbd0e1a9/b91fe581f. Only static
compile/diff checks were run; simulator collection and replay remain
outstanding.

## Route-query batch failure and chunking follow-up (2026-10-24)

A fresh Plan A counterfactual collection at `pickup_scattered/dense`, seed
`3540116847380660224`, source commit `5eb1cb4e0`, attempted output
`...artifacts-20261023-r4/pickup-scattered-dense-3540116847380660224-cf-v2.jsonl`.
Candidate construction aborted at step 0 with
`ValueError: route batch exceeds 64 queries`; no validated output was written.
The candidate builder now validates snapshot identity and duplicate
`(world, slot, candidate_id)` keys across the full route-query list before
submitting consecutive slices of at most `MAX_ROUTE_QUERY_BATCH` (64) to the
existing evaluator. It concatenates results in query order and passes the same
tolerances and learned-map, scale, and provenance arguments to every slice.
The evaluator retains its explicit 64-query cap. Scoped independent review is
pending. Only static compilation and diff checks have run for this follow-up;
no route collection, simulator evaluation, or dashboard replay has validated
it yet.
