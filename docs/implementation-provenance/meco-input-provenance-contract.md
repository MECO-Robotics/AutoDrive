# MECO A/C model input provenance contract

`frc_defense/meco_input_provenance.py` is the canonical, versioned registry for the serialized A1/A2/A3/C1/C2 neural inputs and the ordered C-track candidate feature columns. Its canonical JSON SHA-256 is emitted as `input_contract_sha256` in new training manifests, checkpoints, stats, inference requests/responses, evaluation records, and evaluation reports. Runtime request construction validates the exact tensor-name order against the registry. A registry change therefore changes the digest and is visible in artifacts.

## Track representations

| Track | Serialized model inputs, in order | Representation |
| --- | --- | --- |
| A1 | `raster`, `globals_`, `hidden` | Raster convolutional scorer |
| A2 | `node_features`, `edges`, `edge_mask`, `region_ids`, `globals_`, `coarse_edges`, `coarse_edge_mask`, `hidden` | Hierarchical legal-pose graph scorer |
| A3 | `node_features`, `edges`, `edge_mask`, `region_ids`, `globals_`, `hidden` | Legal-pose graph scorer |
| C1 | `globals_`, `candidate_features`, `hidden` | Candidate-row scorer |
| C2 | `raster`, `globals_`, `candidate_features`, `hidden` | Raster plus candidate-row scorer |

C1 and C2 use identical candidate feature columns and order: 20 named values, followed by the same 20 missingness indicators. Their serialized inputs differ because C2 also consumes the raster. A1/A2/A3 use distinct representations and score candidate rows through their own lookup/scoring paths. The shared observation sources or candidate boundary do not make their representations equal, and this contract makes no fairness or outcome-equivalence claim.

## Source and observation boundaries

The six raster channels are sensed fuel confidence/age, ally position confidence/age from policy localization, and sensed opponent confidence/age. The observation adapter explicitly avoids filling fuel or opponent locations from simulator truth. In curriculum training, focal and teammate locations come from `policy_pose`, which is restored with the Phase 4 delayed sensor frame. Its runtime estimate integrates the simulator's idealized, noiseless `sim.velocity` wheel/gyro odometry feed; it does not reread `sim.pose` after initialization. Hardware qualification must replace that modeled feed with actual onboard odometry/localization telemetry.

The seven global values include own pose, current load, hub timing, and remaining match time. Their sources are mixed: pose is the policy localization estimate, while load ownership and match/hub timing are simulator state. The registry marks the globals as not directly observed and describes this boundary. These simulator-derived scalars must not be described as sensed observations.

The full-field `PerceptionInputs` encoder follows the same pose authority: focal and teammate entity positions and their raster cells use delayed `policy_pose`; its scalar tensor is the same delayed observation passed by curriculum training/inference. Entity velocities and chassis extents remain simulator telemetry, so the representation is not yet a hardware-qualified sensor package.

A2/A3 node features combine raster samples at legal poses with normalized pose coordinates and heading. Legal nodes, directed edges, regions, and coarse transitions are planner-derived from the active grid and footprint-clearance checks; they are not sensor observations.

The C1/C2 candidate builder combines safe action slots, sensor-track identities and confidence/age, field dimensions, own policy localization estimate, sensed opponent tracks, planner-derived routes/ETA, and prior controller commitment. The 20 ordered candidate values are:

`action_index`, `sensor_track_buffer_index`, `goal_x_m`, `goal_y_m`, `heading_bin`, `behavior_index`, `field_length_m`, `field_width_m`, `target_distance_m`, `nearby_sensed_track_count`, `track_confidence`, `track_age_s`, `opponent_distance_m`, `route_time_s`, `legal_node_index`, `route_distance_m`, `risk_penalty_points`, `target_heading_rad`, `is_continue`, `commitment_switch_cost`.

Route feasibility, risk, and commitment fields are derived or mixed-source inputs; their provenance is recorded individually in the registry. The simulator action mask and native legality checks remain authoritative. Route/risk/path costs applied by the deterministic selector outside the neural model are not represented as neural input tensors.

The registry rule forbids simulator truth or simulator-derived state from being labeled observed. It records provenance boundaries, not an assertion that every feature comes from physical sensors.

## Scope

The digest binds schema/order and declared origins, not tensor values, sensor calibration, hardware equivalence, model quality, or a fairness result. This work does not qualify learned policies, establish simulator behavior, or claim that all A/C representations see equal information.
