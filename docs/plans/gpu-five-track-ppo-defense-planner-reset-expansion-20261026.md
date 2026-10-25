# Pass 1 narrow expansion: device-local defense planner cadence across masked resets

**Status:** proposed scope expansion only; no implementation or validation evidence.
**Planning base:** `origin/main` at `d30cdd89b4d404d545254423e9d5bc950804e346`.
**Approved plan lineage:** the snapshot-v2 plan at SHA-256
`f7a9859ee9ad7436f8e9821f6c6ee0beebad37d14bc2c7ac1bb1ce5ef645f94c`,
including its selected-row-reset predecessor
`docs/plans/gpu-five-track-ppo-selected-row-reset-residency-expansion-20261026.md`.
This artifact expands that plan only for the defense planner cadence state below;
all other existing requirements remain unchanged.

## Evidence and reason for expansion

The selected-reset plan explicitly requires an all-false exact no-op and mixed
mask isolation, but its exact authorized target list excludes
`tensor_defense_step.py` and `tensor_defense_opponent.py`. Source inspection
shows those two modules own the global planner fast-path flags that the reset
owner unconditionally changes:

- `TensorDefenseResetMixin.reset_done` writes
  `_planner_tick_aligned = False` and `_planner_full_batch_active = False`
  before it uses `mask`, including for an all-false mask.
- `TensorDefenseStepMixin.step` changes those flags based on the API's
  full-batch/masked-call form.
- `TensorDefenseOpponentMixin._advance_adstar_cadence` consumes those scalar
  flags to choose between a scalar full-batch tick and a device per-world tick;
  the fallback path also clears both flags.
- `_planner_tick` itself is per-world and reset with `torch.where(mask, ...)`.

Thus an all-false reset mutates scheduler state, and a mixed reset cannot
preserve the batch-global fast-path state while expressing different per-world
cadence eligibility. The reset plan's “all-false is a no-op” and
“unselected rows remain unchanged” criteria cannot be met by edits confined to
`tensor_defense_reset.py`. This is a derived-cache/scheduling issue, not a
request to change planner geometry, controls, simulator physics, or reset
semantics.

## Bounded objective and authorized targets

Make defense planner cadence depend only on its per-world device tick state so
that masked resets and masked steps cannot mutate a batch-global scheduling
mode. Preserve the existing full-batch replan cadence and all per-world
planner outputs. The fixed-shape active mask remains on device; no host read of
mask contents or device-derived cadence state is allowed.

| Target | Authorized change | Boundary |
|---|---|---|
| `frc_defense/tensor_defense_reset.py` | Remove reset-time writes to the global alignment/fast-path flags; reset only selected `_planner_tick` entries and any reset-owned per-world cadence state required by the chosen representation. | Retain the existing reset owner inventory and state ordering. No full reset fallback. |
| `frc_defense/tensor_defense_step.py` | Stop mutating batch-global cadence flags from masked/full-batch call form. Pass the existing active mask through the planner cadence path as needed. | No changes to action decoding, planner target construction, integration, rewards, or step semantics. |
| `frc_defense/tensor_defense_opponent.py` | Use the per-world device tick and active mask as the authoritative cadence path; retire or make non-authoritative the scalar fast-path flags for this defense environment. | Replan eligibility must match the existing scalar fast path for aligned full batches and the existing vector fallback for mixed masks. Do not alter AD* geometry, route outputs, or cadence interval. |
| `tests/test_tensor_adstar_cadence_fastpath.py` | Preserve reference equivalence for full batches; add/migrate mixed-mask cadence cases and prove reset does not perturb unselected ticks or the next replan decisions. | Tests only; no weakening of existing cadence assertions. |
| `tests/test_tensor_defense_masked_reset.py` (existing or new) | Assert all-false reset leaves all cadence state bitwise unchanged; all-true matches fresh reset; mixed reset changes selected tick state only and preserves unselected tick/state and next replan schedule. | Include the actual defense environment reset/step call path, with deterministic fixtures. |

No changes are authorized to `tensor_sim.py`, `tensor_3v3.py`, any 3v3 planner
path, planner implementation/kernel, physics, observation, reward, curriculum
state, PPO algorithm, checkpoint schema, training orchestration, or campaign
data. If source tracing proves the scalar flags must be redefined in shared
construction code or used by another environment in a way that cannot remain
compatible, stop and return for another Pass 1 with the newly required exact
owners. The implementation must not modify shared flags in a way that changes
the 3v3 path.

## Required invariants and acceptance evidence

1. `reset_done(mask)` with an all-false mask is an exact no-op for
   `_planner_tick`, cadence eligibility/mode, planner state, and unselected
   observations; no tensor-to-host mask check may be introduced.
2. A mixed reset changes only selected worlds' tick/reset state. Unselected
   worlds retain bitwise-equal cadence state and produce the same subsequent
   replan decisions as a no-reset control with the same active-mask sequence.
3. With full-batch stepping from a fresh reset, the per-world path requests
   replans on exactly the same ticks as the former scalar fast path for several
   replan intervals. Across masked stepping and resets, due masks equal the
   reference per-world modulo schedule.
4. Inactive worlds do not increment cadence ticks or trigger replans. All-true
   reset produces the same initial tick and first replan timing as ordinary
   full reset.
5. The cadence path has no `.item()`, `.tolist()`, `.cpu()`, tensor truth test,
   host reduction, or Python branch controlled by tensor values. Any static
   Python dispatch on whether the caller supplied an active mask must not
   select/reset individual worlds.
6. Planner goals, route commands, action outputs, and physical update behavior
   are unchanged for matched full-batch and masked-step reference cases; only
   the scheduler's implementation/derived mode changes.
7. This expansion does not weaken the inherited reset-owner, per-world RNG,
   optimizer/checkpoint continuation, five-track PPO, replay, or source/version
   identity requirements. Any changed persisted cadence state must be included
   in the existing reset/checkpoint schema contract; no silent checkpoint
   migration is allowed.

## Pass sequence and external gates

1. Pass 1 reviews this exact plan and its source-owner expansion. No
   implementation or runtime work is authorized by this artifact alone.
2. Pass 2 may edit only the listed targets after approval, in a fresh
   HDD-backed worktree based on current main and the approved snapshot-v2 plan
   lineage. Do not edit the active training source/checkpoints or touch device
   leases. If an unlisted owner is needed, stop before editing it.
3. Pass 3 independently reviews exact pushed source and tests against every
   invariant above and the inherited reset/PPO plan requirements.
4. Tests and any device validation remain blocked until the existing plan's
   WX9100 lease-release gate is satisfied. Because cadence changes simulated
   planner behavior, final acceptance also retains the repository's seeded
   end-to-end simulator evaluation, fresh replay inspection, and current
   dashboard backend/frontend checks after Pass 3 and lease release.

This plan records a bounded source-scope correction only. It establishes no
test, GPU, simulator, replay, training, dashboard-currentness, qualification,
or performance result.
