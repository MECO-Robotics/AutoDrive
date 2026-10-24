# Phase 4 policy-pose truth boundary

## Three-pass record

- **Planner:** audited all active 3v3 decision paths for physical-pose reads,
  including observation encoding, `PerceptionInputs`, candidate ranking and
  masks, Plan A roots, Plan C geometry, route starts/obstacles, controller
  targets, and delayed sensor frames. Physics and evaluator-only outcomes stay
  tied to physical truth.
- **Implementer:** `172a665e9c5c2b317a8d87552f7063195266cfa5` adds separate
  `estimated_pose`/`policy_pose` state and routes observations, candidates,
  masks, Plan A/Plan C features, route/controller decisions, and teammate
  obstacle estimates through policy localization. Follow-up
  `7f8c567c0a2a712235a1e464c06b2b3adb1f86b1` routes the active perception
  encoder through `policy_pose` and the delayed observation frame. The
  integrated candidate is `8fe3c9daf411a7914e0ee6bcf7f1fc89ae27b729`.
- **Independent reviewer:** approved exact follow-up head
  `7f8c567c0a2a712235a1e464c06b2b3adb1f86b1` after checking the perception
  encoder, delayed frames, fixed-estimate/changed-truth invariance, remaining
  `sim.pose` uses, and idealized odometry disclosure.

## Evidence and limits

The implementer reports 17 focused tests passing, Python compilation, and
`git diff --check`. Tests cover policy observation/candidate/mask/route
invariance to physical-pose changes at fixed estimates and delayed
`PerceptionInputs`. No end-to-end simulator evaluation or fresh replay was run;
the repository requires both for simulated behavior validation. The pose
mislocalization factor is still not wired to the runtime setter. Modeled
odometry currently integrates noiseless simulator velocity, so it is not
hardware-qualified. Physical truth remains for physics, camera rendering and
occlusion, and labeled evaluator/replay outcomes. No robustness, qualification,
or deployment result is claimed.
