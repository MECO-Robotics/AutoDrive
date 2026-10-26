# Pass 3 review history: PPO/RNG owner response

**Reviewer identity:** `plan_a_owner`
**Reviewer role:** independent Pass 3 reviewer; not the response author
**Reviewed response author:** delegated technical owner agent, `/root/ppo_rng_owner`

## Prior exact head

- Response commit: `88853cc53b055e870c7edf11561be96d78e6d3ed`
- Response path: `docs/plans/v4-owner-responses/ppo-rng.md`
- Response SHA-256: `89167b07be2ccae2e17c16ce8d49f031e2a16f46fc5b624cec1ce153d9c357c7`
- Verdict: **REVISE**

The prior review found that the response did not assign numeric PPO stream
codes or define the draw-site registry, so the adopted key and draw schedule
were not fully reproducible. This verdict applies to that superseded response
head only.

## Corrected exact head

- Response commit: `602565ef99986996d2c12d06024833247dc3d395`
- Response path: `docs/plans/v4-owner-responses/ppo-rng.md`
- Response SHA-256: `219f3466741c17d9e49a4d402f0b3f8ae6f016f0738a068034a8ad312f610d75`
- Verdict: **ACCEPT**

## Scope and evidence checked

I fetched and verified the exact corrected pushed commit and file hash. The
response pins V4 proposal commit
`6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256
`0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`, and
matrix commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256
`9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.
The closure plan's matrix row 57 and PPO/RNG Pass 2 requirements are covered.

The correction resolves the prior issue with stable, separate uint16
registries: PPO stream codes `0=INVALID`, `1=ACTION_CATEGORICAL`,
`2=EPOCH_PERMUTATION`; draw-site codes `0=INVALID`,
`1=ACTION_CATEGORICAL_WORD`, `2=PERMUTATION_FISHER_YATES_WORD`. It states that
these are static source identity and do not add fields or change the V4 field
table/byte layout. It distinguishes these namespaces from
`rng_protocol_code=2` for `meco-gamepiece-philox-v2`, bars gamepiece/physics
use of PPO codes, rejects duplicate assignments/names and per-run key
collisions, makes assigned meanings immutable, and requires a new RNG
domain/version and source identity for changed code meanings, draw sites, or
invocation order. Registry mappings are included in the source digest.

The action and permutation draw schedules are explicit, including canonical
rollout/epoch ordering, rejection sampling and consumed-word counter
advancement, and exclusion of padding. Checkpoint continuation, RNG mismatch
and overflow failure behavior, and no silent fallback to global RNG are
specified. The digest discussion binds code registries and masks while
preserving V4's existing payload layout. These decisions address V4 proposal
lines 122-134 and 287-296 and the matrix's stream derivation/counter and
checkpoint continuation requirements (row 57).

I also rechecked the remaining row decisions and cited source limits: current
terminal/timeout and GAE behavior (`frc_defense/meco_track_training.py:2031-2040,2698-2703,2749-2751`); sampled action/log-probability and executed trace separation, replay, and execution-failure actor exclusion (same file `1961-1969,1997-2000,2104-2115,2128-2149,2156-2162,2173-2195`); candidate sampling (`frc_defense/meco_tracks.py:1681-1689`); and legacy global RNG/checkpoint state (`frc_defense/meco_track_training.py:2255-2260,2322-2365,2815-2832`). The response distinguishes existing behavior from future V4 requirements, including update atomicity (proposal lines 240-249), validity/bootstrap (281-286), and selected/executed action linkage (297-301). Its cross-owner dependencies are appropriately scoped. No tests/runtime were run.

## Disposition and authority

The corrected response's **ADOPT** disposition is supported for the PPO/RNG
owner row. Its author identifies this as a delegated technical decision and
explicitly claims no human maintainer approval, ABI publication, or
implementation authorization. This review ACCEPT does not convert other
PENDING owner rows into adoption and is not human approval or implementation
authorization.
