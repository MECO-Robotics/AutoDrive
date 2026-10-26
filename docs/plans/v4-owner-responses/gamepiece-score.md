# Gamepiece and score owner response for GPU transition ABI V4

Date: 2026-10-26  
Owner role: `gamepiece_score_owner`  
Delegated task identity: `/root/gamepiece_score_owner`  
Response author: delegated technical owner agent

## Pinned inputs and disposition

- V4 proposal: commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256
  `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- V4 owner-adoption matrix: commit
  `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256
  `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.

**Disposition: ADOPT** the exact pinned V4 proposal for the Gamepiece/score
owner row, including its 59-byte typed event row, the fixed 2,304-event batch
capacity, the event code proposal, device-native capture, staged mutation,
reset identity, and fail-closed overflow behavior. This is a delegated
technical decision for this row only.

## Evidence and reasoning

The V4 task-scoped profile uses one world (`N=1`), six robot slots (`S=6`),
32 rollout rows (`R=32`), and 12 physics ticks per macro (`T=12`). The trainer
currently defaults to 12 ticks in `frc_defense/meco_track_training.py:116-125`
and passes that value into `_run_macro_step`; the loop calls `env.step` at
most once per action tick (`frc_defense/meco_track_training.py:1883-1925`).
The simulator's gamepiece update dispatches once through
`TensorThreeVsThreeGamepieceMixin._update_fuel`
(`frc_defense/_tensor_3v3_gamepieces.py:25-28`).

Both active score mutation branches select no more than one piece per robot
per update. The vectorized path computes the first eligible piece with
`argmax`, creates one `chosen` mask, and records its score mutation
(`frc_defense/gamepiece_actions.py:384-429`). The fallback robot loop likewise
uses `argmax` to choose at most one piece for each robot and records that
mutation (`frc_defense/gamepiece_actions.py:439-477`). With six robot slots,
the source-supported maximum is therefore six score events per physics tick.
For a 12-tick macro this is at most `S×T = 6×12 = 72` events per transition;
for 32 rows the batch maximum is `R×S×T = 32×6×12 = 2,304` events. Shortened
macros and inactive robots only lower that bound. The current semantic is one
event per accepted FUEL score mutation, with one score unit per event, as
declared by `frc_defense/meco_simulated_score_ledger.py:24-30`. The adopted
event code registry is `event_type=1` for `fuel_score_awarded`; the source-site
codes are `1=gamepiece_actions.update_fuel_actions.vectorized_score_mutation`
(`frc_defense/gamepiece_actions.py:429`) and
`2=gamepiece_actions.update_fuel_actions.robot_loop_score_mutation`
(`frc_defense/gamepiece_actions.py:476`).

The adopted V4 event record is a packed, ordered 59-byte row with no implicit
padding. The following field order and types reproduce the pinned V4
proposal's “Score-event table and maximum payload” section, lines 195–212:

| Order | Field | Type | Bytes |
|---:|---|---|---:|
| 1 | `valid` | `bool` | 1 |
| 2 | `rollout_row_index` | `uint8` | 1 |
| 3 | `episode_event_sequence` | `uint16` | 2 |
| 4 | `tick` | `int64` | 8 |
| 5 | `match_elapsed_seconds` | `float64` | 8 |
| 6 | `world_slot` | `int64` | 8 |
| 7 | `alliance` | `uint8` | 1 |
| 8 | `robot_slot` | `int64` | 8 |
| 9 | `piece_slot` | `int64` | 8 |
| 10 | `piece_generation` | `int64` | 8 |
| 11 | `event_type` | `uint16` | 2 |
| 12 | `score_units` | `uint16` | 2 |
| 13 | `source_site_code` | `uint16` | 2 |
|  | **Total** |  | **59** |

The total event table is
`2,304×59 = 135,936` bytes. Row index binds an event to one of the batch's 32
transitions; that transition's episode generation disambiguates rows when a
batch spans an episode reset. Event sequence is episode-local. The existing
recorder's maximum is 32,768 events per episode
(`frc_defense/meco_simulated_score_ledger.py:20,412-420`), so its maximum
zero-based sequence is 32,767 and fits the proposed `uint16` sequence field.
This episode recorder bound is distinct from, and does not replace, the
2,304-event profile batch capacity.

The current capture implementation does not satisfy the proposed qualifying
GPU architecture: it clones score masks into Python pending lists and
materializes accepted indices and scalar values on CPU
(`frc_defense/gamepiece_actions.py:13-25,27-76`); training explicitly rejects
the CPU recorder for GPU-qualifying physics (`frc_defense/meco_track_training.py:3921-3935`).
The episode recorder invalidates capture on reset and raises if it is finished
after an invalidation (`frc_defense/_tensor_3v3_gamepieces.py:83-95`,
`frc_defense/meco_simulated_score_ledger.py:409-446`). These facts support
V4's requirements for device-native typed capture, row-plus-generation event
identity, an episode-local sequence, and reset-aware staging; they are not
evidence that V4 capture is already implemented.

The existing score recorder is optional evidence and deliberately catches
recording errors after gameplay mutation
(`frc_defense/gamepiece_actions.py:72-76`). V4 must instead reserve the full
2,304-event table before mutation, stage score events and associated transition
state until validation succeeds, and discard the whole staged decision/update
on overflow or invalid event data. It must not reuse the optional recorder's
post-mutation best-effort behavior. A reset boundary must either finalize the
row under its current episode generation before reset or discard it; sequence
numbers restart only under a new episode generation. No event may be silently
truncated, repaired on the host, or published from a partially committed
transition.

## Scope limits

This adoption covers only the Gamepiece/score row of the pinned V4 proposal.
It does not claim that every simulator roster has the six-slot ThreeVsThree
profile, that another scoring type or site fits these codes, or that V4 is
adopted across other owner rows. A new event type, score rule, source site,
world count, robot count, or macro duration requires an amended immutable ABI
profile and recalculated event capacity and payload size. This response does
not establish implementation authorization, source behavior, runtime
correctness, qualification, or training results.

This response records only this delegated role's technical decision. It is not
human maintainer approval, cross-owner adoption, ABI publication, or
implementation authorization.
