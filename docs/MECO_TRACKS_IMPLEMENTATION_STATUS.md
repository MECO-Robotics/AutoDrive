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
regret or simulator optimality. A separate offline analyzer now reports
replay-verified sampled bounded-set regret for C1/C2 checkpoints; it does not
claim regret over unsampled choices or simulator optimality.
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

## Separate Plan A P independent route-cost oracle v3 (static implementation)

The additive `meco-counterfactual-plan-a-p-route-oracle-v3` contract preserves
the v2 reward-ranked/P0 collector and fitter. The v3 collector records the
planner's frozen base bump-cost grid, the regenerated intervention map and
scale, grid resolution, selected descriptor, and every executed route-trace
snapshot. A separate pure-Python segment/grid integrator scores each snapshot
in normalized grid-edge units from the declared base bump grid plus the exact
effective P map. It never reads AD* g/f/path costs or simulator returns. Labels
are preferences between sampled executed paths only; lower independent cost
is preferred, exact ties are omitted, and the labels do not establish route
optimality or per-cell supervision.

`collect-plan-a-p-routes-v3` emits the new schema. The existing strict branch
replay materializer regenerates and re-integrates each cost grid and rejects
any mismatch before fitting. `warm-start-plan-a-p-v3` trains only on non-tied
oracle cost comparisons. Its prediction is the independently replayed base
bump-grid integral plus the learned scaled P-map integral; both terms and the
oracle labels use normalized grid-edge units. The base term is recomputed from
the strict replay's frozen grid and each archived path trace, then added once to
the differentiable P term. This keeps the fixed base route cost from being
omitted or counted twice. The command and function require an available explicit
CUDA device. The weights-only artifact stamps oracle version, dataset and route
provenance, and sampled executed-path coverage. This tranche has
received static compilation and diff checks only. No P-oracle collection,
fit, simulator evaluation, or dashboard replay has run; runtime qualification
and exhaustive legal-grid coverage remain outstanding.

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

## Route-query batch failure and chunking follow-up (2026-10-23)

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
The evaluator retains its explicit 64-query cap. Independent scoped review approved isolated commit 1f9f7d079 and integration
commit f23171d63. Only static compilation and diff checks have run for this follow-up;
no route collection, simulator evaluation, or dashboard replay has validated
it yet.

A fresh retry at `pickup_scattered/dense`, seed `3540116847380660224`, after
the route-query chunking fix failed strict prefix replay parity at
`snapshot_id` inside `_branch_return` (r5 attempt); no validated output was
written. `_prefix_snapshot` had independently rebuilt safe action candidates,
which advanced the episode identity separately from the route-feasible
candidate snapshot. It now takes the safe observation from
`build_route_feasible_candidates` and uses that single snapshot for the Plan A
support ID, replay `snapshot_id`, and `safe_observation_sha256`. Existing strict
replay checks and digest fields are unchanged; no schema bump is needed. This
correction passed independent scoped review in isolated commit b627f3a3b and
was integrated as 0379776b7. Static compile/diff/JSON checks only; collection,
simulator evaluation, and dashboard replay remain outstanding.

A fresh retry at `pickup_scattered/dense`, seed `3540116847380660224`,
source integration commit `72fe1059c`, attempted output
`...artifacts-20261023-r6/pickup-scattered-dense-3540116847380660224-cf-v2.jsonl`
and failed in `_branch_return` with `RuntimeError: strict prefix replay parity
mismatch at simulator_state_sha256`; no validated file was written. Repeating
the same-seed reset comparison produced equal global RNG hashes, while
`_state_value` differed only at `_physics_graph_command.tensor`. The graph
command and active-mask buffers were uninitialized `torch.empty` scratch; both
are copied from commands/active masks before any captured graph replay. They
were zero-initialized in `0c65857f3` to retain strict digest parity. The
follow-up GPU run exposed that the root environment's `_physics_graphs` field
contains device execution cache objects, whose identities are not simulator
state and cannot be canonically replayed. State digest schema
`meco-full-environment-state-v2` omits only this exact root attribute from
traversal; all other environment fields and global RNG state remain hashed
strictly, and unknown state continues to fail closed. Both dataset manifest
validation and warm-start artifact validation require schema v2, so v1 artifacts
cannot be reused under this contract. Scoped implementer change is isolated at
`codex/meco-cudagraph-digest-v2` at `a4ff27922`, approved by the independent
reviewer and integrated as `07098e82a`; static compile/diff/JSON checks only.
Fresh GPU collection and simulator replay remain pending.

## HIP broadphase scratch-state digest follow-up (2026-10-23)

Both GPU retries in `r10` failed before writing validated datasets during
`_rebuild_and_check` with `RuntimeError: strict prefix replay parity mismatch
at simulator_state_sha256`. They used source integration commit
`881bf4b3d`, scenario `pickup_scattered/dense`, seed
`3540116847380660228` on `cuda:0` and seed `3540116847380660232` on
`cuda:1`, with attempted outputs
`/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive-meco-final-plan-artifacts-20261023-r10/gpu0/pickup-scattered-dense-3540116847380660228-cf-v2.jsonl`
and
`/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive-meco-final-plan-artifacts-20261023-r10/gpu1/pickup-scattered-dense-3540116847380660232-cf-v2.jsonl`.
For the same-device reset comparison on seed `3540116847380660228`,
observation, candidate, support, and global RNG digests matched; the only
reported state differences were `env.fuel_physics._hip.contacts.tensor`,
`_hip.list.tensor`, `_hip.next.tensor`, and `_hip.robot_candidates.tensor`.
`FuelHIP.__init__` allocated those four execution buffers with
`torch.empty`. The scoped follow-up initializes only those four to zeros,
retaining their shapes, devices, and dtypes. `rebuild` still starts at one, so
the first HIP substep rebuilds/initializes the broadphase buffers before valid
reads; the buffers remain included in the strict state digest and its schema is
unchanged. The implementer change `dea6ec05763` passed independent scoped
review and was integrated as `92ea05e9a`. Static compilation, diff, and JSON
validation only; no simulator run succeeded, no dataset was validated, and
simulator/replay validation remains pending.

## D0 calibration GPU and margin provenance implementation (2026-10-23)

The calibration implementation now requires an explicit visible GPU device
(`cuda:N`), records its runtime identity, and rejects CPU-backed calibration
reports during bundle intake. Calibrated v2 gain margins now derive from an
operator-declared per-category operational gain threshold plus a declared
multiplier times the maximum calibration stratum standard error; the standard
error uses both between-seed and within-seed repeat variance. The desired gain
therefore remains explicit and is not inferred from noise. The frozen v2
protocol records calibration report hashes and the complete gain derivation;
promotion report evidence binds those fields. Existing frozen v1 construction
is unchanged. Deterministic, resumable episode sharding and strict merge into the existing
v2 report shape are now implemented in a separate scoped pass. Shards bind exact
repeat/case/seed work units, source/runtime/configuration and case-table identity;
merge rejects missing/duplicate/unexpected work, provenance or safety drift,
recomputes summaries from raw rows, and keeps protocol intake as the sole
acceptance gate. GPU runtime, merged-report intake and protocol creation remain
unvalidated.

This implementer pass received static review only. No calibration, simulation,
training, test suite, or dashboard evaluation was run, so GPU runtime behavior,
calibration output acceptance, and the end-to-end report/protocol chain remain
unproven. Static compilation and diff checks are the only planned checks for
this worktree.

## D0 calibration sharding tranche v1

Planner and implementation are complete in isolated worktree
`codex/meco-d0-calibration-shards` at commits `8d18502c5` and `7c1ba15fa`. The
source identity hash was corrected to match the evaluator's existing
serialization. Scoped review approved feature commit `7c1ba15fa` and integrated
HEAD `997a04aaa`. Shards are immutable, resumable, deterministically assigned
and merge only after exact tuple coverage and normalized runtime checks. No
calibration, GPU collection, simulator evaluation, or protocol creation was
run. Runtime evidence remains pending.

## GPU-only defaults and snapshot capture follow-up (2026-10-23)

In isolated worktree `codex/meco-gpu-only-freshness` based on runtime integration
`755633d3`, observation snapshots now capture host monotonic time at acquisition
before simulator tensor reads or CPU serialization, while preserving simulation
time separately. Supervisor CLI and API validate all training and evaluator
assignments as explicitly indexed, available CUDA/ROCm devices before workers
can launch. The standalone trainer, supervised collection and warm-start,
counterfactual collection, Plan A route collection, and Plan A P warm-start
paths likewise reject omitted, CPU, unindexed, or unavailable devices. Qualification
documentation now gives the target/prerequisite calibration bundle and margin
sequence before v2 validation and identifies v1 as historical and nonpromotable.

Static review only: no tests, simulator evaluation, calibration, GPU runtime,
or dashboard replay were run. Qualification, evaluation, and promotion evidence
remain pending. The GPU-only contract applies to simulator/model/evaluator
compute; orchestration, tensor serialization, checkpoint map-location, and RNG
handling may still use the host CPU.

## §4 asynchronous versioned inference contract (implementation pass)

The production A and C evaluator paths now use `meco-inference-v2` through a
per-adapter asynchronous broker. Requests bind the model ID, checkpoint hash,
protocol ID, adapter generation and monotonic request sequence, observation
snapshot, captured/dispatch/completion/expiry times, and canonical input order
and shapes. C requests bind their ordered candidate IDs and feasibility-mask
digest. Plan A requests bind the exact legal graph node/pose/behavior support
digest and include sensed-target descriptor IDs in the canonical candidate
ordering. The single pending slot replaces an older queued request; the worker
rejects expired requests and publishes only complete validated responses.
Action-boundary consumers verify all echoed identity and digest fields and
reject late or superseded responses. A and C paths compute their deterministic
fallback while inference runs, then poll once without waiting before action
commit; an unavailable result is discarded for that snapshot. The worker uses
`torch.inference_mode()`, and sequence acceptance plus queue replacement are
serialized under one broker lock. Decisions expose request provenance and
dropped, late, and inference-fault counters. Existing feasibility, route/risk,
deterministic fallback, and Plan A P path-cost behavior are preserved; there is
no CPU model fallback. `evaluate_meco` scopes every constructed adapter and
closes workers in `finally`, including failures during partial checkpoint
loading. Worker close is idempotent, rejects new requests, discards queued
work, sends a stop sentinel, and joins before evaluator-owned models are
released. Accepted Plan C provenance includes the simulation observation time.

Three-pass ledger: Planner complete; implementer complete at `f25a9f596`;
scoped static review approved the integrated candidate at `2318d696e`. Static
checks only (`py_compile`, `git diff --check`); no tests or simulator evaluation
were run. Worker lifecycle, queue replacement, stale/out-of-order rejection,
and fallback behavior still require runtime and replay validation.

## Plan A legal-node supervision tranche v1

A separate strict replay-bound dataset and fitter now cover a deterministic
sample of legal Plan A nodes, preserving the existing v3 dataset and weights
schemas. Each branch binds full canonical node order, reachable mask, ordered
pair feasibility layout and support digest. The collector applies the archived
legal support action, follows the source's teacher continuation, horizon, and
per-macro discount, and records two paired repeat outcomes. It never uses
AD* internal costs as supervision. U is the equal-weight mean over feasible
behavior means after descriptor sampling; B preferences use same-node paired
logit contrasts, with unavailable and censored feasible behaviors and ties
explicit. U is withheld if any feasible behavior outcome is censored;
independently valid B pairs remain usable. P remains null until independent
P-v3 labels exist.

The tranche is sampled rather than exhaustive. It selects at most
`min(floor(N/16), floor(512/(2*B)))` nodes per branch, where `N` is the count
of feasible reachable nodes and `B` is the feasible behavior count, with a
heading-bin × graph-region deterministic selector and feasible root inclusion.
One uniformly hash-ranked descriptor is sampled per selected node/behavior,
with inclusion probability and inverse-probability weight recorded. The hard
budget is 512 simulator branches per prefix and 12,000 for the dataset; the
preflight upper bound is source row count times 512 and rejects oversized
source files before collection. At the approximate 7,528-node/five-behavior
support, the node tranche is at most 51 nodes (about 0.68%).

Three-pass ledger: planner complete; implementation committed as `f91d76237`;
scoped static review approved the exact diff. Runtime evidence pending. Static `py_compile`, diff whitespace,
and JSON parsing are the only checks performed here. No tests or simulator
runs were run.

## Joint Plan A U+B+P warm start v1

A new standalone fitter combines replay-bound legal-node U/B supervision and
independent oracle-v3 executed-route P supervision in one model and optimizer,
so encoder gradients are shared rather than spliced from separate checkpoints.
It cross-validates exact trajectory-v3 source data and manifest hashes across
both dataset manifests, plus protocol, identity, train split, prediction mode,
and indexed device. Its weights-only schema binds both datasets/manifests,
records explicit equal task weights and loss masks, and requires `p_mode=learned`
with the exact P scale in training. Existing U+B and P-only artifact schemas
remain unchanged. Implementation committed as `15d5ef218`; scoped static review approved the
integrated candidate at `05338ce72`. Runtime evidence remains pending. Static
checks only; no tests, collection, fitting, GPU training or simulator runs
were performed.

## Sampled counterfactual regret analysis

The offline `scripts/meco_counterfactual_regret.py` analyzer scores the exact
replayed Plan C candidate table with a source-bound C1/C2 checkpoint. It
reconstructs the model's recurrent state through the verified teacher prefix,
then replays every sampled candidate continuation and checks its macro,
horizon, discounted, continuation-policy, and terminal outcomes before using
the returns. Regret is `max sampled available gross_horizon_return` minus the
checkpoint-selected candidate's return, and is emitted only when that exact
candidate ID has a sampled outcome. Missing chosen outcomes are excluded and
counted. Stable table order resolves score ties.

This is sampled bounded-set regret, not regret over unsampled choices or global
simulator optimality. D1/D2 regret is unavailable because each v2 archived
table carries one prediction-mode risk tensor; the analyzer does not present
the same tensor as two independent controls. The script is outside the
`frc_defense` source-identity tree and therefore does not change existing
training artifact identities. Scoped review approved implementation commit
`ba7b65b0`; integrated review and runtime replay remain pending. Static
`py_compile` and diff checks only; no tests, simulator, or GPU execution were
performed.
