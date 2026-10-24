# Phase 4 context runtime module

## Three-pass record

- **Planner:** specified full-team start profiles, deterministic teammate
  behavior assignments, observation-only focal-pose offsets, and robot-to-robot
  contact exposures. It defined contact realization as physics-observed and
  required recovery to mean return to the pre-contact route-following state.
- **Implementer:** `19737bd51429ae41346a3aa7e4435d00d6d943bb` adds
  `frc_defense/meco_phase4_context.py` and focused tests. It maps three runtime
  start profiles to exact registry IDs/config digests, uses canonical
  `hold_position`, defines distinct contact geometries, requires a fresh sensed
  opponent track targeting the focal robot, and records contact/clearance only
  as explicitly labeled proxies. Registry contact and route-following recovery
  signals remain unavailable. Its observation transform updates dependent
  channels but reports `training_no_leak_complete: false` because upstream
  candidate, mask, route-root, and physical-pose readers are not redirected.
  The combined candidate reviewed for API consistency is
  `f2867715c08e71fd6365b8ddb5ba742e62109fba`.
- **Independent reviewer:** approved the exposed APIs on exact combined
  candidate head `f2867715c08e71fd6365b8ddb5ba742e62109fba`, after checking
  contact geometry distinctions, fresh-target gating, config/registry
  identity, and honest proxy/no-leak limitations.

## Evidence and limits

Sixteen focused tests passed, with Python compilation and `git diff --check`.
No simulator, GPU, replay, or dashboard evaluation was run. The module is not
wired into curriculum resets, teammate evaluation, contact metric ledgers, or
the campaign runner. The post-observation transform is not a complete
no-truth-leak boundary. No contact, recovery, pose robustness, qualification,
or deployment result is claimed.
