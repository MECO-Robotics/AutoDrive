# MECO track adapter status

`frc_defense/meco_tracks.py` contains isolated experimental model and scoring
interfaces for the five tracks in the [evaluation plan](MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md).
These modules do not change the production controller or make any promotion or
deployment claim.

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

The eight-row source is an integration seam for the current curriculum
simulator, not the plan's general feasible candidate generator. It does not
enumerate arbitrary goal poses or behaviors, compute AD* route ETA, guarantee
capture-pose clearance, or measure candidate recall against an oracle. Its
`route_time_s` is unavailable and route costs are `null`; the current adapter
must not claim route-cost parity or subtract a fabricated cost. Existing action
masks and executors still define which action can be selected.

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

The module does not yet provide the plan's end-to-end matched PPO experiment,
supervised warm start, paired held-out evaluation suite, D0 production
controller variance, Hailo/ONNX export or hardware profiling. The document's
baseline freezes, defender-family splits, route-costed candidate recall,
full-game metrics, simulator records, and confidence-interval decision gates
remain required before any performance conclusion.

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
