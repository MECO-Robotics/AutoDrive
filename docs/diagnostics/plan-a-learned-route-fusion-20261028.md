# Experimental learned-cost route fusion

The experimental `AUTODRIVE_EXPERIMENTAL_LEARNED_ROUTE_FUSION=1` switch lets the existing HIP Bellman and lattice extraction kernel handle learned traversal costs. It keeps learned paths unshortcutted. The default remains the prior learned-cost route.

A guarded `pickup_simple/single` seed 71 comparison on an exclusively leased WX 9100 used a real 12,096-node pose graph with a synthetic support mask. The unfused warm route stage took 1.431 s; fused calls took 0.611 and 0.584 s. The first unfused call took 20.214 s with initialization. All route fields, masks, fallback and route evidence digest matched exactly. This is one warm baseline, not an end-to-end training measurement.

Four guarded HIP planner parity tests passed for zero, gradient, barrier and near-tie learned maps with multiple starts, headings, robot footprints, obstacles and an inactive row. These timings and tests used the same runtime operations but predate making the switch opt-in. A subsequent guard stopped the worker during scenario initialization at its configured limit, before ten repeat timing pairs ran. A fresh simulator replay, current dashboard verification, repeated paired timing, and final-source validation remain open. Logs are under `AutoDriveDiagnostics/learned-route-fusion-20261028/`.
