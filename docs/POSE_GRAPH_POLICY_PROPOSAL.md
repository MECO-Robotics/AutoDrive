# Proposed pose-graph policy for 2026 REBUILT

**Status:** Design draft. None of the observation, graph, policy, or action contracts below is implemented by this document.

## Objective

Train a policy that uses simulated perception to choose its own FUEL collection routes and respond to defense throughout a match. Keep scoring and ferrying as predetermined commands: `score` aims at the alliance HUB and shoots, while `ferry` aims toward the alliance zone and shoots. The policy chooses when to issue those commands and how to move. It must select `score` only while its HUB is active and `ferry` only while its HUB is inactive.

The policy has two planning scales. A shared graph network scores destinations and travel costs over the whole field. A route search turns those scores into a coherent path. A local, mode-aware decoder converts the next route segment into a drivetrain command whose predicted swept motion remains in modeled legal space.

## Coordinate systems and static geometry

- Use field coordinates for the persistent map and route graph. Normalize robot position by field length and width. Express drivetrain translation and tracked motion relative to the robot when constructing local control features.
- Store the 2026 field geometry, scoring regions, alliance zone, and legal transition topology once. Mirror or transform this geometry into a consistent alliance frame before building policy inputs.
- Build the robot's configuration space from its actual footprint. A configuration is a planar pose $q=(x,y,r)$, where $r$ is heading. A position can be legal at one heading and illegal at another.
- Discretize field position into cells $i$ and heading into periodic bins $k$. A pose node $(i,k)$ exists only when the robot fits there. A directed edge exists only when a feasible translation and rotation between its endpoints has a collision-free *swept footprint*. Nearby poses separated by a wall have no edge. In-place turns are checked as swept rotations.
- Cache the field geometry and graph construction for a given robot footprint and resolution. Robot size can vary, so do not assume one pose graph is valid for every robot. Retain continuous $x,y,r$ for control and collision checks; discretization is for planning.

The input map remains two-dimensional. Heading-dependent pose nodes and edges are derived from static geometry, rather than duplicating the changing perception map across every heading bin.

## Observation

At each policy step, rasterize retained FUEL, ally, and opponent tracks onto the field cells. Object footprints occupy all cells they cover; radius is therefore represented spatially rather than as another channel. The tracker predicts possible future occupancy along physically reachable transitions. It must not paint a predicted path through a fixed obstacle.

Keep a cell if the robot center can legally occupy it at least one heading; remove the other cells from the policy tensor. The proposed dynamic map has shape `[N_cells, 3]`. An object footprint that overlaps a removed cell must still affect adjacent reachable cells when it changes collision or collection opportunities:

| Feature | Meaning |
| --- | --- |
| `class` | `empty`, `FUEL`, `ally`, or `opponent`; encode categorically inside the network rather than treating the IDs as ordered numbers. |
| `confidence` | For `empty`, confidence that the cell is clear; for an object class, confidence in that current or predicted occupancy. Low confidence distinguishes unobserved space from known free space. |
| `signed_time` | Zero for current occupancy, positive for predicted arrival, negative for a stale last observation. Normalize to a fixed tracking/lookahead horizon. Use a reserved value for empty cells with no predicted arrival. |

When several objects or arrival times overlap one cell, a single class and time cannot represent them all. The first version needs a documented priority rule, such as earliest relevant robot arrival before FUEL, and should record whether this compression loses important defense information. Confidence decay and spatial spread are computed by the tracker; confidence alone is not treated as a velocity measurement.

Supply a separate 10-value robot and game vector:

```
[x, y, sin(r), cos(r), v_forward, v_lateral, omega,
 estimated_FUEL_load, signed_HUB_timer, match_time_left]
```

Normalize positions, velocities, angular velocity, load, and times using fixed physical scales. The signed HUB timer is positive for active time remaining and negative for inactive time remaining; at an exact zero boundary, allow neither shooting command until the new phase is known. Robot $x,y$ remain in this small vector because the network must relate map objects and routes to the robot's current location. The heading pair avoids angle wraparound.

Static obstacles are **removed from the changing feature tensor**, not from the policy's knowledge. Fixed node coordinates, legal pose states, edge connectivity, clearance, and destination labels provide that knowledge. A conventional CNN over the packed rows would invent neighbors across removed cells. Use graph message passing over legal transitions, with relative geometry supplied as derived features.

## Global policy and route output

Lift each 2D cell feature into the legal pose nodes associated with that cell. Add derived relative position and pose heading features without repeating them in the external observation. A shared graph encoder produces pose embeddings. Condition it on the robot/game vector early enough for it to distinguish a nearby defender from the same defender across the field.

Shared output heads produce:

1. A score for each eligible goal pose or goal region. Candidate goals include reachable FUEL, useful search regions, scoring positions, and ferry positions. Scoring and ferry goals use their fixed field locations and heading requirements.
2. A positive cost for each legal pose transition. Static travel time and clearance may be fixed edge features; the learned part reflects dynamic occupancy, timing, and tactical value. Fixed obstacles never become traversable because of a low predicted cost.
3. Scores for `none`, `score`, and `ferry` command modes. The phase and load requirements are enforced by the command interface, independently of learned scores.

Retain the selected goal across policy steps. Search the legal pose graph using current predicted edge costs to get a complete route, then execute its first segment. Replan the route as defense changes; change the retained goal when it becomes unavailable or another choice clears a defined improvement threshold. This keeps route choices global while allowing the model to learn which defenders and FUEL opportunities matter. The route search adds computation but no trainable parameters.

The output scores are variable in count: one per eligible pose goal and one per legal edge. Shared heads keep the number of trainable weights nearly independent of graph size. Node and edge count still govern runtime and memory use.

## Local output transform

The planner's next segment constrains the local motion choice. At current pose $q$, candidate mode $m$, and a short control horizon, construct a set of controls whose **entire predicted trajectory** respects the robot footprint, drivetrain limits, static geometry, and currently modeled dynamic occupancy. Generate and certify connected safe regions of this control space; the model scores only the regions actually constructed. If none is available, the decoder commands a stop and requests replanning.

For `none`, a local region can be a certified tetrahedron in `(v_forward, v_lateral, omega)` space. The model emits three unconstrained values, transforms them into four nonnegative barycentric weights summing to one, and combines the tetrahedron's four vertices. Every decoded control lies inside that region. Certifying only the vertices is insufficient: the full region and its swept trajectories must be valid. For `score` and `ferry`, automatic aiming owns `omega`; construct safe regions over the remaining translational controls using the **actual aiming rotation** in the trajectory check. The decoder returns physical drivetrain commands and the selected command mode.

This transform replaces arbitrary raw velocity output and after-the-fact clipping. It also differs from a next-waypoint output: each selected region represents a legal short motion, including its path and rotation. The global route supplies long-range intent. Unknown or poorly tracked opponents cannot be guaranteed absent; dynamic avoidance quality depends on the tracker and prediction horizon.

## Model size and relation to current code

The present `BaselineMLP` uses 137 scalar inputs, two 128-unit layers, eight categorical action outputs, and a value head: **35,337 trainable parameters**. The present full-field architecture uses a 12-channel, 32×64 raster plus the 137-value scalar vector. Neither implements this proposal.

A small graph design with two 64-wide message-passing layers, a 64-wide robot-state branch, a 128-wide fused representation, shared goal/edge/region scorers, mode and local-coordinate heads, and a value head is **roughly 50,000 trainable parameters**. This is an illustrative estimate, not a measured count: heading features, class embedding, message-passing depth, and mode-conditioned region scoring are not yet fixed. Static graph construction, route search, and the geometric output decoder add no trainable weights. Their runtime cost must be measured separately. Flattening `[N_cells, 3]` into the existing MLP would make its first layer grow with field resolution and lose legal adjacency.

## Implementation boundary and evaluation

This proposal requires new observation building, a footprint-aware pose graph, graph-policy and route heads, a mode-aware safe-region generator, and an action decoder. The current eight-tactic interface and 137-scalar checkpoints are incompatible with the new input and output contracts; existing checkpoints cannot be resumed into this architecture without an explicit transfer design. Existing score/ferry routines can supply aiming behavior, but their phase and motion interfaces need to be checked against the proposed contract.

Start with a fixed map resolution and heading-bin count, then measure graph construction, policy inference, route search, and safe-region generation separately. Compare the new policy with the current deterministic and learned baselines on held-out defenders, starting positions, FUEL layouts, and phase schedules. Report acquisition, active-HUB scoring, inactive-HUB ferrying, collisions, illegal transitions, route completion, goal changes, and time per decision. Representative end-to-end simulator evaluations must exercise the changed behavior and inspect a fresh record or replay, with scenario and reproducible seed retained. If a dashboard replay is used, first restart and verify the current backend and frontend stack as required by `AGENTS.md`.

## Decisions to settle before implementation

- Field cell size and heading-bin resolution, including how to handle localization between cells and headings.
- Whether all FUEL and defender information needed for a decision survives one class and one signed time per cell; overlapping occupants may require a richer representation.
- The track decay, forecast horizon, and meaning of confidence for observed free cells versus predicted occupancy.
- The goal set, goal-retention rule, and whether learned edge costs need explicit static travel cost and clearance terms.
- A tractable method for constructing and certifying continuous safe control regions at the policy rate, especially while automatic score/ferry aiming changes rotation.
