# Pass 3 review: PPO/RNG owner response

**Reviewer identity:** `plan_a_owner`
**Reviewer role:** independent Pass 3 reviewer; not the response author
**Reviewed response author:** delegated technical owner agent, `/root/ppo_rng_owner`
**Response commit:** `88853cc53b055e870c7edf11561be96d78e6d3ed`
**Response path:** `docs/plans/v4-owner-responses/ppo-rng.md`
**Response SHA-256:** `89167b07be2ccae2e17c16ce8d49f031e2a16f46fc5b624cec1ce153d9c357c7`
**Verdict:** **REVISE**

## Scope and evidence checked

I verified the exact pushed response commit and its file SHA-256 against the
accepted closure plan's PPO/RNG row and pinned inputs. The response pins the
V4 proposal at commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256
`0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`, and
the owner matrix at commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`,
SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.
The accepted closure plan requires the PPO/RNG owner to resolve row validity,
bootstrap, atomicity, action linkage, Philox derivation/counters, checkpoint
continuation, digest encoding and failure behavior, while leaving unsupported
choices as explicit gates (matrix row at lines 57; closure-plan Pass 2 table).

The response covers terminal/truncation and GAE, actor/critic validity,
sampled-action/log-probability linkage, update and RNG-counter atomicity,
Philox draw schedules, checkpoint continuation, digest fields and failure
handling. Its citations support the described legacy implementation and
limits: trainer end/timeout and GAE behavior (`frc_defense/meco_track_training.py:2031-2040,2698-2703,2749-2751`), replay/action linkage and execution-failure actor exclusion (same file `1961-1969,1997-2000,2104-2115,2128-2149,2156-2162,2173-2195`), sampling contract (`frc_defense/meco_tracks.py:1681-1689`), and legacy RNG seeding/checkpoint state (`frc_defense/meco_track_training.py:2255-2260,2322-2365,2815-2832`). The response correctly distinguishes those source facts from future V4 behavior and says no tests were run.

I checked the pinned V4 recommendations on masks/bootstrap (lines 281-286),
Philox derivation and counter behavior (287-293), digest framing (294-296),
action separation (297-301), and whole-update failure atomicity (240-249), as
well as the response's cross-owner dependencies. Its explicit no-human-approval,
no-publication, and no-implementation-authority statements are appropriate.

## Reasons for REVISE

The response's **ADOPT** disposition is not supported as written because its
PPO RNG identity is still incomplete. It says the action and permutation
stream codes are distinct and covered by source/profile identity, but does
not assign their numeric code values. It also refers to a draw-site registry
being included in the source digest without defining that registry. V4 says
draw-site IDs and key-derivation code are source-bound (proposal lines
122-134), and its RNG recommendation derives each stream key from the stream
code (287-293); therefore these omitted values determine the keys and are
needed to reproduce and checkpoint the stream state. The matrix expressly
requires stream derivation/counters and continuation evidence (row 57).

The owner should either supply an immutable, explicit PPO action/permutation
stream-code and draw-site registry decision, including the exact values and
ordering used in key/draw derivation, or change the disposition to **PENDING**
and name those fields as unresolved. The response can then be reviewed at a
new exact head. This review does not reject its other row decisions, and does
not substitute reviewer judgment for the owner's missing technical decision.

## Disposition and authority

The response is in the correct row scope and its stated source limits,
cross-owner boundaries, and lack of human-approval or implementation claims
are supported. Its current **ADOPT** disposition is not supported until the
stream-code and draw-site identity fields above are resolved. This record is
an independent delegated review only; it is not human maintainer approval,
shared ABI publication, or implementation authorization.
