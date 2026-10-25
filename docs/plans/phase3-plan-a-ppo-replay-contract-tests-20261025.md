# Phase 3 Plan A PPO replay contract tests

## Planner pass

**Status:** proposed offline test tranche; no implementation or behavior claim.

**Planning base:** `origin/main` at `fd9c0d3eec7a0802eb6ef95a384efa4c594357d8`.
The approved Phase 2 branch head `d20a55a18259c4dc41e01b8040c8c2826ebc6833`
changes the multi-case coordinator and training manifest, but leaves the Phase
3 PPO and policy files unchanged from this base.

**Source requirement:** `docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`
§5 Phase 3 requires an actual stochastic policy distribution over feasible
Plan A choices and valid log probabilities. Existing code implements this
support and replay check, but the current PPO replay regression tests exercise
Plan C only. This tranche adds direct, offline regression coverage for the
Plan A path across A1, A2, and A3.

## Current code anchors

- `frc_defense/meco_plan_a_policy.py:21-42` defines frozen node × behavior
  support; `:103-112` recomputes its digest.
- `frc_defense/meco_track_training.py:1075-1107` forms Plan A policy logits
  from utility, behavior, route, risk, and optional P costs, then masks invalid
  support rows.
- `frc_defense/meco_track_training.py:1668-1679` reconstructs Plan A logits
  for PPO; `:1693-1727` verifies support digest and old log probability before
  optimizer work.
- `frc_defense/meco_track_training.py:1767-1811` excludes failed execution
  rows from actor loss while retaining critic loss.
- `frc_defense/meco_tracks.py:1399-1424` defines masked categorical sampling
  and returns sampled action, log probability, and entropy.
- `tests/test_meco_c1_ppo_replay.py` covers C1 frozen-input replay/mutation;
  `tests/test_meco_tracks.py:242-257` covers generic masked sampling. Neither
  exercises Plan A PPO replay for A1/A2/A3.

## Scope for Pass 2

Add one focused pure-tensor test module, `tests/test_meco_plan_a_ppo_replay.py`.
Construct a small synthetic graph and `PlanAActionSupport` fixture directly;
do not create a simulator environment or call graph search. Use actual A1/A2/A3
model forwards and the existing `_policy_inputs_for_row` / `_ppo_update` seams.
Keep changes test-only unless the tests expose a production defect; any such
fix must be proposed as a separate reviewed scope.

Required cases, parameterized over A1/A2/A3:

1. **Masked stochastic support:** with at least one valid and one invalid
   descriptor, every sampled action indexes a valid support row; invalid rows
   have zero probability. The sampled old log probability is finite and equals
   the distribution's log probability for that exact support index.
2. **Unchanged rollout replay:** save model inputs, graph tensors, support,
   route/risk node costs, hidden state, selected support index, and log
   probability; `_ppo_update` must reconstruct the same probability and pass
   the pre-update guard.
3. **Mutation rejection before optimizer step:** independently mutate a
   policy-relevant input and a digest-bound support tensor. Each must fail
   before `optimizer.step` is called. The input mutation must cause a replay
   log-probability mismatch; the support mutation must fail its recomputed
   digest. Use a counting optimizer or equivalent assertion to prove there was
   no partial update.
4. **Critic-only execution failure:** a row marked as remasked/fallback or
   otherwise execution-failed contributes no actor loss or entropy bonus, but
   remains available for value loss. Assert the update diagnostics reflect
   zero actor contribution for the failure row.

Use fixed random seeds and deterministic small tensors. Keep the learning rate
zero or use an injected optimizer where needed to observe validation without
changing the fixture between cases. Do not weaken existing tolerances or skip
one of A1/A2/A3 to make the suite pass.

## Pass 2 acceptance criteria

- All four cases pass for each of A1, A2, and A3 on CPU using synthetic tensor
  inputs only.
- The successful replay uses the same exact frozen support index and
  recomputed distribution; no invalid support item is sampled.
- Both mutations are rejected before any optimizer step.
- Failed execution remains critic-only; it receives no policy-gradient or
  entropy contribution.
- Existing focused `tests/test_meco_c1_ppo_replay.py` and
  `tests/test_meco_tracks.py` remain green.
- `py_compile` and `git diff --check` pass. Record exact tested commit and
  command results; request independent exact-head review before integration.

## Limits and exclusions

This tranche does not run `build_plan_a_support`, simulator environments,
training, GPU/HIP code, campaign data, or dashboard services. It checks the
mathematical PPO input/log-probability contract with constructed support only.
It does not establish that sampled actions execute safely in simulation, that
the reward/risk objective improves outcomes, or that A1/A2/A3 perform well.
Any behavior test later must follow the repository's `AGENTS.md` simulator and
fresh-dashboard replay requirements. Phase 3 performance, matched budgets,
Phase 2 warm-start consumption, and qualification remain separate gates.
