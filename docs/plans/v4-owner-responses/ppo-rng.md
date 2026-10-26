# PPO/RNG owner response for shared command and transition ABI V4

**Owner role:** `ppo_rng_owner`  
**Delegated agent/task identity:** `/root/ppo_rng_owner`  
**Response author:** delegated technical owner agent  
**Response date:** 2026-10-26  
**Disposition:** **ADOPT** the exact pinned V4 proposal for the PPO/RNG owner row, with the row-local decisions below resolving its underspecified mechanics. These are delegated technical decisions, not human approval.

## Pinned artifacts

- V4 proposal, commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256
  `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- V4 owner-adoption matrix, commit
  `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256
  `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.

## Source facts

The trainer currently reduces environment end and timeout to one `done`
(`frc_defense/meco_track_training.py:2698-2703`). `_gae` uses that same mask
for the value delta and recurrent advantage (`2031-2040`), and the caller
supplies a zero bootstrap when either condition ends the last row
(`2749-2751`). Thus current training does not implement V4's distinct
truncation bootstrap.

The PPO rollout stores categorical `action_index` and its sampled `logprob`
(`frc_defense/meco_track_training.py:2691-2703`). Replay and PPO loss evaluate
that same index against frozen/recomputed logits
(`2104-2115,2128-2149,2187-2195`). The semantic high-level action is stored
separately as `executed_high_level_action`; per-tick simulator outcomes are
recorded in `executed_action_trace` (`1961-1969,1997-2000,2691-2703`). Execution
failure excludes the row from actor loss but retains critic learning
(`2156-2162,2173-2186`). Candidate sampling asserts finite legal support and
returns the selected category's log probability from the same distribution
(`frc_defense/meco_tracks.py:1681-1689`). The existing masked-sampling and
replay tests encode matching action/log-probability and exact replay
expectations (`tests/test_meco_tracks.py:275-290`,
`tests/test_meco_c1_ppo_replay.py:41-50,54-89`). These citations describe
source/test contracts; no tests were run for this response.

Current action sampling and epoch permutations use implicit PyTorch RNG state
(`frc_defense/meco_tracks.py:1681-1689`,
`frc_defense/meco_track_training.py:2164-2167`). Training seeds Torch and
Python globally (`2255-2260`), checkpoint schema v1 stores Torch CPU, Python,
and optional device RNG state (`2815-2832`), and resume restores those states
(`2322-2325,2354-2365`). These facts establish the legacy continuation
mechanism, not V4 Philox behavior or bit-identical cross-version continuation.

## Adopted row decisions

### Validity, bootstrap, and action linkage

Adopt V4's distinct `row_valid`, `actor_valid`, and `critic_valid` masks and
the lifecycle rule in proposal lines 281–286. For this row, the mask truth
table is:

| Condition | `row_valid` | `actor_valid` | `critic_valid` | Bootstrap |
|---|---:|---:|---:|---|
| Padded/unused slot | 0 | 0 | 0 | none; no GAE, replay, RNG, optimizer, or diagnostics effect |
| Valid transition with legal selected policy action and valid old log probability | 1 | 1 | 1 when observation/reward/value are valid | according to terminal/truncation below |
| Valid transition with execution failure/remask/preemption/fallback | 1 | 0 | 1 when observation/reward/value are valid | none |
| True terminal | 1 | as otherwise valid | as otherwise valid | none; terminal takes precedence if both terminal and truncated bits are set |
| Truncation without true terminal | 1 | as otherwise valid | as otherwise valid | bootstrap only from a valid final next observation |
| Nonterminal continuation | 1 | as otherwise valid | as otherwise valid | valid next value required at the rollout boundary |
| Empty/invalid policy support or invalid required row data | fail the complete decision/update | 0 | 0 | none; no fallback action may be credited as a sampled action |

`transition_finished` is true for terminal or truncated transitions. GAE may
use the final observation value to compute the truncation row's delta, but
must stop its cross-row recurrence at either terminal or truncation so the
next episode cannot leak backward across a reset. Terminal wins if both flags
are true. A failed execution has no bootstrap even when the simulator episode
continues; its valid reward/value data may still train the critic only, as the
current source does. Invalid or nonfinite observation, reward, or value makes
the row invalid rather than being converted to zero.

The sampled `policy_local_index`/`source_global_row` and its exact sampled old
log probability remain paired. `action_id`/semantic selected command and the
fixed per-tick `executed_action` trace are separate fields. Native remasking,
fallback, preemption, target loss, or zero executed ticks never changes the
sampled index or recomputes its old log probability. Such execution failure
sets actor validity false; the critic may consume the transition if its
observation, reward, and value are valid. This preserves the current actor
skip/critic-only behavior while retaining an auditable sampled-versus-executed
record. The map from each family-local index to the command identity remains
owned by Plan A/Plan C and the shared-command producer; this response does not
change their ordering or IDs.

### Update atomicity and failures

Adopt the V4 whole-decision/update no-op rule for every validation, digest,
capacity, schema, replay, or RNG failure (proposal lines 240–249, 281–296).
No failed attempt may commit partial model parameters, optimizer moments or
step counters, or action/permutation RNG counters. The implementation must
validate the entire fixed batch before mutation and stage the update, or retain
enough state to restore all affected model, optimizer, and RNG state before
reporting failure. The PPO transaction's commit boundary includes all epochs
and rows of one update; a later-row failure cannot leave earlier-row Adam
steps committed. For a full rollout/update transaction, snapshot both stream
counters before the first decision in that batch and commit their new values
with the optimizer/model update only after the complete batch succeeds. A
failure after simulator execution stops the worker; the physical world is not
claimed to roll back. Recovery resumes from the last committed checkpoint and
reconstructs the environment from its frozen task seed instead of retrying the
already executed transition. This adopts the required outcome, not a specific
implementation.

Current `_ppo_update` performs `optimizer.step()` in its row loop
(`frc_defense/meco_track_training.py:2164-2200`) after replay validation
(`2076-2150`); it does not establish rollback for a failure after an earlier
step. V4's update transaction is therefore a future contract requirement, not
a claim about current behavior. Simulator physics cannot generally be rolled
back after a tick. Any simulator/native failure that must prevent a command or
score-event mutation must be preflighted or staged by the command, observation,
and gamepiece owners before mutation; PPO failure handling cannot retroactively
undo the external world. This is a cross-row boundary, not an expansion of this
owner's authority.

### Philox streams, checkpoint continuation, and digest bytes

Adopt proposal lines 122–134 and 287–296. The row-local interpretation is:

1. The action and permutation streams are independent Philox4x32-10 streams.
   Their stream codes are distinct and covered by the source/profile identity.
   `rng_protocol_code=2` continues to identify the separately specified
   `meco-gamepiece-philox-v2` protocol; it is not the PPO stream algorithm
   identifier. The action/permutation stream codes and draw-site registry are
   included in the source digest as V4 requires.
2. Derive each 64-bit key from SHA-256 over the exact V4 domain string
   `meco-ppo-v4-rng-v1`, root seed as uint64 little-endian, track code, and
   that stream's code; interpret the first eight digest bytes as a
   little-endian key. Each stream has its own 128-bit counter. Serialize each
   uint32 word little-endian in the canonical payload; counter advancement is
   the number of consumed 32-bit Philox words, including any rejected words.
   Overflow fails closed and cannot wrap.
3. The action stream draws once for every valid stochastic policy decision
   with nonempty support, in rollout-row order, before execution outcomes are
   known. Its categorical transform uses action support's canonical order and
   one uniform draw `u=(word+0.5)/2^32`, computed in float64. Convert finite
   float32 logits to float64, subtract the maximum, exponentiate, and normalize
   with a fixed ascending-index sum over legal support. Select the first legal
   category whose cumulative normalized probability is strictly greater than
   `u`; the final legal category absorbs any cumulative rounding remainder.
   This transform, including the math mode, is versioned in the source digest.
   A failed/padded invalid decision consumes no action draw. The permutation
   stream uses descending Fisher–Yates over valid rollout rows in ascending
   epoch order. For each `i` from `n-1` to `1`, draw uint32 words until
   `word < floor(2^32/(i+1))*(i+1)`, then set `j=word mod (i+1)` and swap
   positions `i,j`; all rejected words advance the counter. Critic-only
   execution-failure rows remain in the permutation because they have valid
   transitions. Padded rows are excluded.
4. The derived keys and counters are continuous for a run across episode and
   chunk resets; resets do not reseed either PPO stream. Fresh runs start both
   counters at zero. Checkpoints persist root seed, track/stream codes, exact
   counters, protocol/source/profile identity, and optimizer/model state as
   one continuation point. Resume restores counters before any stochastic PPO
   draw and rejects absent, malformed, overflowed, or incompatible RNG state;
   it never silently reseeds or falls back to CPU/PyTorch global RNG. A saved
   state after a failed update is the pre-rollout transaction state, per
   atomicity above. Bit-exact continuation is required only for the same pinned
   source, profile, backend, device identity, and checkpoint state; a different
   identity fails compatibility checks instead of claiming cross-device bit
   identity.
5. For `input_digest`, compute SHA-256 over ASCII tag
   `MECO4V4/PPO_INPUT\0`, then exact canonical packed bytes for the profile,
   source, config, and checkpoint digests; track code and code registry;
   `row_valid`, `actor_valid`, `critic_valid`, `command_valid`, and reset mask;
   observation and recurrent-before fields; and the applicable immutable
   policy-support/candidate identity digest. `support_digest` remains over the
   exact family-specific support/candidate content and ordering under the
   Plan A/Plan C owner's adopted contract, and includes the V4 profile digest,
   row masks, and applicable code registries. For `transition_digest`, compute
   SHA-256 over ASCII tag `MECO4V4/PPO_TRANSITION\0`, then `input_digest`,
   canonical row identity/generation and rollout/update/epoch/sample-order
   fields; selected policy index and command identity; exact old
   log-probability, entropy, reward, value, old-value, return, and advantage
   bytes; terminal/truncation/bootstrap, actor/critic/command/transition
   validity fields; fixed macro trace; recurrent-after state; and both
   streams' keys and before/after counters. Integer and float fields use V4
   little-endian encodings, bools are one byte 0/1, and array order is the
   field-table order; no JSON or host re-encoding participates. Each of the
   three row digests includes the profile digest, row masks, and applicable
   code registries. Hashes are computed on device. The bundle payload digest
   covers the exact full payload bytes. Host persistence copies opaque bytes
   only.

The current checkpoint's schema v1 Torch/Python RNG fields
(`frc_defense/meco_track_training.py:2322-2365,2815-2832`) are not a compatible
substitute for this V4 state. Artifact/manifest must bind the new RNG protocol
and continuation fields to the checkpoint version; changing them invalidates
resume compatibility. The exact row layout and canonical digest definitions
must be reflected in the reconciled ABI's byte totals and the artifact owner
must include the corresponding checkpoint/profile provenance. These are
cross-row consequences of this adoption, not decisions for the artifact owner.

## Scope and remaining dependencies

This response adopts the exact V4 SHA for the PPO/RNG row, including the
row-local semantics above. The current trainer does not yet implement these
requirements; adoption is not implementation evidence. Plan A and Plan C
owners retain their support ordering and source-row identity decisions; the
environment/profile owner retains seed/profile identity; the artifact owner
retains checkpoint schema/provenance and byte envelope; the observation/native
owner retains command preflight and external-state mutation; gamepiece/score
retains score-event staging; evaluation/curriculum retains the product
estimand. Those dependencies do not qualify another row or imply any owner's
consent. This response records only the delegated `ppo_rng_owner` technical
decision and is not human maintainer approval, shared ABI publication,
repository-governance approval, or implementation authorization.
