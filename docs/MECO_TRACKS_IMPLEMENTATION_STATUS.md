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

## Deliberate limits

The candidate service now exposes K=8/16/32 tables from eight existing action
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

A1 scores a dense field but this slice has no qualified legal-pose lattice or
SE(2) transition mask. The A1 adapter can gather scores only for candidate
rows that supply explicit goal coordinates, heading bins, and behavior IDs;
the caller remains responsible for supplying actually feasible poses. A2 and
A3 require an explicit deterministic legal graph (`node_features`, directed
`edges`, `edge_mask`, and `region_ids`); they fail when that graph is absent.
The repository's current XY AD* search and experimental XY route probe are not
silently treated as this interface.

The Plan A traversal penalty field is returned separately and is not treated
as an endpoint penalty. A deterministic planner must integrate it over a
verified route before the Plan A policy is considered operational. Candidate
ranking currently uses dense utility and behavior scores only.

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
