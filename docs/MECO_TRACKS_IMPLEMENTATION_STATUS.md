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
  and a parent tree. The support mask/order and selected pair descriptor are
  retained for PPO log-probability recomputation. `candidate_k` applies only to
  Plan C.

## Deliberate limits

The Plan C/D1/D2 candidate service exposes K=8/16/32 tables from eight existing action
primitives, with stable collection-offset rows, a current-snapshot AD* route
query, ETA/risk features, and selected XY endpoint override. Its recall report
is explicitly within this same pool; K32 is not an independent or larger
oracle. It does not enumerate arbitrary behaviors or measure recall against a
distinct >32 reference set. Static endpoint clearance encloses the robot and
extended intake at every yaw. A sensed-target bearing supplies a nonbinding
`capture_estimate`; it is not a feasibility mask because the native controller
does not command or guarantee the inferred terminal heading. Dynamic future
swept-robot collision remains outside this candidate feasibility check, and
the simulator's native intake/action executor remains authoritative.

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
robot arbitration. Intake support binds a sensed target slot and a
bearing-aligned capture envelope; score support is restricted to clear score
grid endpoints; ferry support requires carried fuel and the current ferry
endpoint; defense support is restricted to the current intercept/post endpoint.
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
supervised warm start, Hailo/ONNX export, and hardware profiling remain open.

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
