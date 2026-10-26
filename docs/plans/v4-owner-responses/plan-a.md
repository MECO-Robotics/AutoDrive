# Plan A owner response for PPO transition ABI V4

- **Owner role:** `plan_a_owner`
- **Delegated agent/task:** `/root/plan_a_owner`
- **Response author:** delegated technical decision owner, `/root/plan_a_owner`
- **Response date:** 2026-10-26
- **Disposition:** **ADOPT** the exact pinned V4 proposal for the Plan A owner row.

## Pinned artifacts

- V4 proposal: commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256
  `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- V4 owner-adoption matrix: commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`,
  SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.

This response records the Plan A row only. It adopts the proposed Plan A
capacities, field inventory/order, and digest binding as written in the pinned
proposal; it does not revise the V4 byte table.

## Decision and source evidence

I adopt the padded pose graph capacity `A_N=12,096` with runtime legal nodes
selected by the capacity-sized `node_valid` mask. The source builds capacity
as eight headings times the active planner's `nx × ny`, stably compacts legal
poses into a valid prefix, and retains the fixed capacity in `node_valid`
([`meco_pose_graph.py:20-31,141-179`](../../../frc_defense/meco_pose_graph.py:20)).
The V4 value is specific to the proposed 56×27×8 profile; it does not assert
that all 12,096 entries are legal. The graph's planner resolution and grid
dimensions derive from the active simulation and planner configuration
([`tensor_adstar.py:257-284`](../../../frc_defense/tensor_adstar.py:257)).

I adopt eight ordered Plan A behavior/descriptor rows per padded node, giving
`A_G=96,768`. The current support construction creates the fixed eight-column
layout and flattens it in node-major order ([`meco_plan_a_policy.py:359-378`](../../../frc_defense/meco_plan_a_policy.py:359)). I adopt ten candidate edge slots per
padded node: the graph lays out eight translation slots followed by two
heading-turn slots, with validity carried separately by `edge_mask`
([`meco_pose_graph.py:181-209`](../../../frc_defense/meco_pose_graph.py:181)).

I adopt 14 ancestor levels. The source starts with the parent table and
repeatedly doubles its covered span until it reaches node capacity
([`meco_plan_a_policy.py:393-399`](../../../frc_defense/meco_plan_a_policy.py:393)); for 12,096 entries this produces 14 levels. The V4 field
ordering is the listed Plan A sequence: support mask; distance, route cost,
parent-edge length; parent node and traversal order; ancestor jumps; node
utility; behavior logits. The source support structure has corresponding
ordered support/search arrays ([`meco_plan_a_policy.py:28-47`](../../../frc_defense/meco_plan_a_policy.py:28)) and emits the eight descriptor rows in canonical node-major order
([`meco_plan_a_policy.py:359-378`](../../../frc_defense/meco_plan_a_policy.py:359)).

I adopt float64 storage for the three V4 recorded cost columns
(`distance_m`, `route_cost`, `parent_edge_m`). This is a serialization choice
for V4 replay evidence. Current graph geometry uses `env.sim.pose.dtype`, and
the current support builder casts these costs to `graph.xy.dtype`
([`meco_pose_graph.py:157-162`](../../../frc_defense/meco_pose_graph.py:157),
[`meco_plan_a_policy.py:381-415`](../../../frc_defense/meco_plan_a_policy.py:381)).
The promoted values therefore preserve current computed values in a wider
recording representation; this decision does not claim that current planning
recomputes costs in float64 or that source float32 values gain additional
precision.

I adopt the proposed digest boundary: profile identity binds the fixed Plan A
graph profile and `node_valid`; the Plan A support digest binds the ordered
support/search evidence and snapshot identity; the transition/payload digest
binds the complete canonical packed payload, including the Plan A model-output
columns. The current runtime support digest already binds ordered support and
graph tensors, including `node_valid`, graph edges/masks, and graph status
([`meco_plan_a_policy.py:84-112`](../../../frc_defense/meco_plan_a_policy.py:84)); its
header includes snapshot and support-order identities and it hashes the
canonical tensor sequence ([`meco_plan_a_policy.py:115-146`](../../../frc_defense/meco_plan_a_policy.py:115)). The CPU reference uses the same
support tensor inventory ([`meco_plan_a_policy.py:149-165`](../../../frc_defense/meco_plan_a_policy.py:149)). V4's canonical field-order, row-mask,
code-registry, profile-digest, and device-produced digest rules are specified
in the proposal's digest section. An implementation plan must freeze the
exact byte encoding and make every digest reproducible from the pinned V4
profile; this response does not claim that the existing runtime digest is
already byte-compatible with V4.

## Limits

This adoption is limited to the Plan A owner row and exact V4 proposal hash.
The cited source supports the shape, ordering, current dtype, and existing
support-digest inputs. The 56×27×8 capacity is profile-specific, while legal
node count remains determined by `node_valid`. No hardware memory admission,
device performance, runtime parity, simulator behavior, or qualification
result is established here. Those require separate reviewed implementation
and validation work.

This is a delegated agent's technical decision under the scoped owner-response
assignment. It is not human maintainer approval, cross-owner adoption, shared
ABI publication, or implementation authorization.
