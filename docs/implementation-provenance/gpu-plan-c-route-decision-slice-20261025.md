# Plan C route-decision implementation provenance

This note records the source audit performed before implementation of the
approved device-resident Plan C route-decision slice.

## Audited source references

- Implementation base: `dd6ca02af38d998eec8f956645de88065c19f7b0`
- Approved scope plan: `docs/plans/gpu-resident-route-decision-slice-20261025.md`
- Weighted route-cost reference: `8372b58de47489949f7c13b4cc20c489b718b9a2`
- Candidate batch numeric reference: `01a000f231e8f5bb79497091094054c5b8f4dd8f`
- Candidate fallback/provenance follow-up, audited but not imported:
  `4b6bf3fbbb53b43524a54325a365a0e10bdff191`

## Reuse decision

From `8372b58de`, implementation may reuse the numerical method in
`_path_weighted_cost`: compute each segment's intersections with raster x/y
boundaries, sort the crossing parameters, sample the terrain multiplier at
each interval midpoint, and weight by segment length in normalized grid-edge
units. The source hunk is not imported as-is. It converts points and costs to
float64, then calls `.item()` and returns a Python `float`; the new path must
retain padded batched paths, masks, and outputs on the route device.

From `01a000f23`, implementation may reuse audited formulas from
`_endpoint_clear_batch`, `_goal_nodes_batch`, `_candidate_geometry_batch`, and
`_forecast_risk_batch` where they preserve current Plan C geometry, stable
candidate ordering, feature meaning, and risk semantics. These formulas are
not accepted wholesale. The candidate-batch branch still builds Python route
queries and consumes `RouteResult` objects, extracts per-candidate values via
`.cpu().tolist()` and scalar `.item()` calls, reconstructs feature rows on the
host, and only then rebuilds tensors. Those host-synchronizing sections are
rejected for the live decision path.

The follow-up `4b6bf3f` changes endpoint fallback handling and provenance;
it is not the candidate-batching implementation. Its code is not imported as a
shortcut. Existing fail-closed fallback semantics remain required and will be
preserved directly in the scoped implementation.

## New implementation paths

The scoped implementation will add typed `DeviceRouteBatch` and
`DeviceCandidateBatch` contracts, a device-returning route evaluator with an
explicit post-decision `RouteResult` projection, device-resident candidate
feature/missingness encoding and feasibility/risk/route-cost tensors, vector
D1/D2 scoring and selection, and one compact selected-action handoff. The
intended source paths are:

- `frc_defense/experimental_route_evaluation.py`
- `frc_defense/meco_candidates.py`
- `frc_defense/meco_tracks.py` only if its current table adapter cannot accept
  already encoded tensors
- `frc_defense/meco_track_training.py`
- focused tests in the files named by the approved plan

No Plan A, AD* kernel, training schedule, checkpoint, campaign, dashboard,
qualification, simulator, or device behavior is included in this audit or
implementation scope.
