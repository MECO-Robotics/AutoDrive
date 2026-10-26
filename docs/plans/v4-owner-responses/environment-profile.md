# Environment/profile owner response: proposed ABI V4

- **Owner role:** `environment_profile_owner`
- **Delegated agent/task identity:** `/root/environment_profile_owner`
- **Response author:** delegated technical decision owner
- **Response date:** 2026-10-26
- **Pinned V4 proposal:** commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`
- **Pinned owner-adoption matrix:** commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`

## Disposition

**ADOPT** the exact pinned V4 proposal for the requested MECO ThreeVsThree five-track training profile, with one track code bound for each run's lifetime. This is a profile decision for this training objective; it does not make the profile universal to other environments or establish production qualification.

| Field | Adopted V4 value | Evidence and scope |
|---|---|---|
| Profile ID | `meco-ppo-v3-3v3-f504-n1-s6-r32-t12-v1` | Pinned V4 §“Proposed task-scoped profile”, lines 42–46. |
| Worlds `N` | 1 | Trainer creates one curriculum environment per training task (`frc_defense/meco_track_training.py:2357-2358`). |
| Robot slots `S` | 6 | ThreeVsThree exposes six robot assignments and world/robot-shaped outputs (`frc_defense/tensor_3v3.py:37-42`); target arrays include six robot slots (`:368-378`). |
| Rollout rows `R` | 32 | `MecoTrainConfig.rollout_steps` default (`frc_defense/meco_track_training.py:116-125`). This adopts the profile default, not an invariant for every configurable run. |
| Macro ticks `T` | 12 | `MecoTrainConfig.action_ticks` default (`frc_defense/meco_track_training.py:116-125`). This is the profile cadence, not a maximum enforced by the config validator. |
| Target/fuel slots `E` | 504 | ThreeVsThree constructor default is 504 and validation permits up to 600 (`frc_defense/tensor_3v3.py:45-48,72-78`); runtime target tables use `(world, six robots, fuel_count, ...)` axes (`:368-378`). Adopt 504 for this profile only. |
| Plan C width `K` | 16 | Trainer default and allowed set (`frc_defense/meco_track_training.py:139,154-155`); the V4 proposal uses a separate continuation slot (`V4 §“Proposed task-scoped profile”, lines 54-59`). |
| Field | 16.54 m × 8.07 m, `2026_rebuilt` | ThreeVsThree constructor defaults (`frc_defense/tensor_3v3.py:63-66`). |
| Observation raster | `[6,25,51]`, 0.3302 m cells, profile-encoded float32 | Cell size is `RASTER_CELL_METERS` (`frc_defense/learning_inputs.py:12`); raster height/width are ceiling(field width/length divided by cell size), six channels are allocated (`:73-78`). For the stated field this gives 25 × 51. V4 fixes float32 in its profile declaration (lines 56, 149-150); the current adapter allocates using `policy_pose.dtype`, so runtime schema implementation must explicitly enforce the adopted float32 encoding. |
| Planner grid | 56 × 27 cells at 0.3 m; 8 headings in the Plan A padded lattice | Planner defaults to 0.3 m and computes `nx=ceil(field_length/resolution)`, `ny=ceil(field_width/resolution)` (`frc_defense/tensor_adstar.py:257,271-284`); pose graph is 8-heading (`frc_defense/meco_pose_graph.py:142-162`). This is geometry evidence only; Plan A capacities remain the Plan A owner's separate decision. |
| Track binding | One run binds exactly one of `1=A1, 2=A2, 3=A3, 4=C1, 5=C2`; zero is invalid | Trainer config has one `track` and validates it against `MECO_TRACKS` (`frc_defense/meco_track_training.py:116-120,142-145`). The pinned proposal specifies lifetime binding and codes in lines 64-67. The five codes map the requested five approaches; this decision does not combine multiple tracks into one run. |

The proposed defaults align with the current MECO trainer and ThreeVsThree environment. They remain explicit profile values because source configuration allows variation: in particular, fuel count may reach 600, rollout length and action ticks are configurable, and candidate K may be 8, 16, or 32. A run serialized under this profile must bind the adopted values and reject a mismatch rather than silently treating those other configurations as this profile.

## Scope limits

This response decides only the environment/profile matrix row. It does not decide Plan A search storage or dtype, Plan C feature ordering, route capacities, target-handle lifecycle, RNG/PPO semantics, score events, payload publication, curriculum qualification, or implementation details. The grid dimensions are reported as profile geometry evidence; Plan A remains responsible for adopting its own graph/search fields. No source, schema, or runtime change is authorized by this response.

This records only the delegated environment/profile technical decision under the accepted response plan. It is not human maintainer approval, cross-owner adoption, ABI publication, production qualification, or implementation authorization.
