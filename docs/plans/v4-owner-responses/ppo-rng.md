# PPO/RNG owner response for shared command and transition ABI V4

**Owner role:** `ppo_rng_owner`  
**Delegated agent/task identity:** `/root/ppo_rng_owner`  
**Response author:** delegated technical owner agent  
**Response date:** 2026-10-26  
**Disposition:** **PENDING**

## Pinned artifacts

- V4 proposal, commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256
  `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- V4 owner-adoption matrix, commit
  `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256
  `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.

## Evidence and decision

The V4 row inventory separates `row_valid`, `actor_valid`, and `critic_valid`
and records terminal/truncation/bootstrap flags (proposal §“Exact dynamic row
field inventory”, lines 147–160). Its lifecycle recommendation says padded
rows have no GAE, replay, RNG, optimizer, or diagnostic effect; terminal and
execution-failure rows do not bootstrap; truncation bootstraps only with a
valid next observation (lines 281–286). The field inventory provides a useful
starting point, but it does not completely define how actor/critic masks
interact with a terminal or truncated transition, a missing next observation,
or an invalid rollout slot. The status field also has no explicit row-level
status code for each of those combinations.

Current training stores a single `done` value as `ended or timeout`
(`frc_defense/meco_track_training.py:2698-2703`), and `_gae` uses that one mask
for both value bootstrap and advantage recurrence
(`frc_defense/meco_track_training.py:2031-2040`). The caller supplies zero
bootstrap for either terminal or truncation (`frc_defense/meco_track_training.py:2749-2751`).
Thus current behavior does not implement V4's proposed truncation bootstrap;
the proposal needs an explicit transition and recurrence rule before its
semantics can be adopted as a compatible successor contract.

The current PPO path records the sampled policy index and its old log
probability, then replays the probability at that same `row["action"]`
(`frc_defense/meco_track_training.py:2104-2115,2128-2149,2187-2195`). Separately,
macro execution records the per-tick `env.last_actions` trace
(`frc_defense/meco_track_training.py:1961-1969,1997-2000`) and the rollout
stores `executed_high_level_action` beside the policy `action`
(`frc_defense/meco_track_training.py:2691-2703`). This supports retaining
selected and executed actions as separate facts. However, V4 does not specify
the mapping when a macro has multiple executed actions, is remasked, preempted,
or fails before a tick, nor how `command_valid`, actor validity, and old
log-probability validity combine in those cases (proposal lines 147–160 and
297–301). The actor-skipped execution-failure path is currently critic-only
(`frc_defense/meco_track_training.py:2156-2162,2173-2186`); the V4 actor/critic
mask truth table should explicitly preserve or intentionally revise that rule.

V4 proposes Philox4x32-10, SHA-256 key derivation, 32-bit-word counter
advancement, separate action/permutation streams, counter overflow failure,
and checkpoint continuation (proposal lines 287–293; the matrix requests
stream key/counter/draw schedule and checkpoint state at matrix line 57).
Current action sampling calls `Categorical.sample()` without a dedicated
generator (`frc_defense/meco_tracks.py:1681-1689`), and PPO shuffling calls
`torch.randperm` without one (`frc_defense/meco_track_training.py:2164-2167`).
Initialization seeds Torch and Python globally (`frc_defense/meco_track_training.py:2255-2260`);
resume restores Torch CPU, Python, and device RNG state
(`frc_defense/meco_track_training.py:2322-2325,2354-2365`), and checkpoint
serialization writes these states (`frc_defense/meco_track_training.py:2815-2832`).
The V4 proposal is a future replacement, not a description of this source.

The proposal's row table records per-stream `key:uint32[2]`,
`counter_before:uint32[4]`, and `counter_after:uint32[4]` for two streams
(lines 157–159), while its bundle prefix stores a root seed, segment ID, world
UID, and one `rng_protocol_code`; the prose derives each key from root seed,
track code, and stream code (lines 122–134, 287–293). It does not say whether
row keys repeat the bundle key or identify derived keys, how each row's
counters relate to the batch stream, or how segment IDs advance and restore
across chunk resets, episode resets, and resumed checkpoints. It also says
`rng_protocol_code=2` binds `meco-gamepiece-philox-v2` for the bundle prefix
(lines 125–134), without defining whether that code covers the distinct PPO
action and permutation Philox protocol. These are contract ambiguities, not
merely differences from current implementation.

V4 requires canonical field-order packed-byte SHA-256 digests and device-made
digest bytes (proposal lines 294–296), but does not identify the exact digest
domain and per-digest byte spans for the PPO row, action/log-probability pair,
RNG state, and full batch. The generic envelope digest rules do not settle
which dynamic PPO fields are covered by each `input_digest`,
`transition_digest`, or `support_digest` column. Without those boundaries a
producer and validator could both follow the text and compute different yet
internally consistent digests.

I therefore leave this row **PENDING**. To resolve it, publish an immutable
successor or addendum that pins: (1) a complete validity/status truth table and
GAE equations for terminal, truncation, execution failure, and invalid/padded
rows; (2) selected action, each executed tick action, and old-log-probability
validity/linkage for remask, preemption, and zero-tick failure; (3) exact
Philox stream key/counter ownership, draw schedule and word consumption,
counter/segment advancement, reset behavior, and checkpoint resume state; (4)
separate protocol codes or an explicit shared registry for gamepiece versus
PPO RNG; and (5) canonical digest domains and byte spans for all PPO/RNG
identity fields. Then reconcile the affected row and dependent artifact/profile
sizes before adoption.

This response records only the delegated `ppo_rng_owner` technical decision.
It is not human maintainer approval, cross-owner adoption, shared ABI
publication, or implementation authorization.
