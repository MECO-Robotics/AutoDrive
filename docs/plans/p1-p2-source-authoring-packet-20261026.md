# P1/P2 source-authoring readiness packet

**Date:** 2026-10-26  
**Purpose:** Pin the first permitted source-only slice after P0a acceptance and identify the P1/P2 gates. This is planning documentation only; it does not authorize source edits, validation, or integration.

## Decision

No P1 or P2 source-only slice is authorized yet. P0a is accepted, but the accepted overall plan requires each package to have its own finite Pass 1-accepted package plan, named distinct implementer and reviewer, an exact base, and accepted dependency heads. The accepted P0a checkout contains no scoped P1 package plan/allowlist and no P1 source candidate with an independent SOURCE/API ACCEPT record.

The first source-only slice is **P1's initial score-event/schema source-trace and implementation slice**, on the accepted implementation base `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`. Its exact file allowlist must be set by a scoped P1 package plan and accepted in Pass 1. Start only after that plan's exact-head Pass 1 ACCEPT and assignment of distinct Pass 2 and Pass 3 roles. The overall plan's P1 scope is schema/serialization modules, source-mapped score-event producer/consumer paths, artifact/manifest code, and schema fixtures; it is not a sufficient finite file allowlist by itself.

G0 does not gate P1/P2 source authoring or static review. It does gate builds, test execution, numerical/runtime work, simulator/profiler, device/GPU access, dashboard work, training, checkpoint mutation, integration, merge, and promotion. The active campaign and device leases remain outside this packet's authority.

## Package gates

| Package | Source-only status | Exact start gate | Interface dependency / blocked work |
|---|---|---|---|
| P1 — V4 schema, score events, envelope and artifacts | **Not started.** P0a and the overall plan are accepted; scoped P1 Pass 1 plan and source/API candidate review are absent from the accepted P0a head. | A finite P1 package plan receives independent Pass 1 ACCEPT; distinct implementer/reviewer are named; worktree branches from `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`. Then P1 may perform source tracing and author only its accepted allowlist, with static review. | Before P1 source freeze, bind the approved profile and P0a score semantics; verify the exact source producer/consumer map. P1's exact candidate API must receive independent pre-G0 SOURCE/API ACCEPT before P2 source authoring can start. This disposition is not package-final acceptance or integration permission. |
| P2 — GPU-resident environment, observation, candidate and route core | **Blocked.** The current P2 package document records its G0/source-authoring amendment as REVISE and says corrected successor/fresh Pass 1 review are pending. | First, independent exact-head Pass 1 ACCEPT of the corrected P2 package plan; assign distinct implementer/reviewer. Also require the P1 candidate API/source head's independent SOURCE/API ACCEPT. Then branch from the same approved source base and observe P2's finite allowlist. | No P1 branch merge/copy as a P2 base before G0. P2 source freeze can provide an API pin to later package authors, but neither package may integrate or runtime-use it before G0 and final reviews. |

## Exact evidence pins

- Accepted overall plan: commit `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`; plan SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; Pass 1 ACCEPT record commit `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`.
- Accepted P0a candidate: commit `56183cefd2402342ce28f836155164fc850ebc60`; file SHA-256 `c8ace83e0e5a3fdef8d33ab090d3cbb9193e12342b015d72b07016acdb603726`.
- P0a maintainer decision record (user reply “Approve all”): commit `7007607010636fe7451ad1b59eb1a900e0a4d7b1`; file SHA-256 `d9394b07bc508021a4b10faf89b2d258a863183288d633442f796fc181592e64`.
- P0a exact-head Pass 3 ACCEPT record: reviewer commit `d5dab585ef1147f0d9997ece80db6e9a7ba95979`; record SHA-256 `25a2c38c502599a4da9c9448bdd3e6850b3ae7145d5977870e17b98304c280e6`.
- Approved implementation source base: `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.
- P2 current amended-plan branch: `codex/v4-p2-gpu-residency-plan-20261026`, head `fa15258776030ccb2ca5f9f637a38594075797d7`; P2 plan file SHA-256 `123ff2f8741636eabc1aea4b876a59ac326fa81f4739a469ecc4271f4d6807ca`. Its status text says the predecessor amendment received REVISE and a corrected successor/fresh review are pending. The accepted overall-plan P2 scope is not a substitute for this scoped-plan gate.
- Reconciled profile V4 envelope digest: `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`; reconciled artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`.

## Existing score-event plan assessment

The existing `gpu-resident-score-event-buffer` plan (`codex/gpu-score-event-buffer-plan-20261026`, head `47ff1037dc5bf0035b5a1d4523f5bddae240351e`, file `docs/plans/gpu-resident-score-event-buffer-20261026.md`, SHA-256 `3095cbee304f21d1d84ae91c1152722f955df47fba58f2399d3e6302d518eff1`) is useful prior design evidence but cannot become the P1 package plan through a status/pin amendment alone. It targets the legacy episode recorder cap of 32,768 accepted events and a passive report drain. The reconciled P1 ABI instead requires a distinct fixed 2,304 × 59-byte batch table, full-capacity staging before mutation, and all-or-nothing transition failure. The existing plan also requires editing `frc_defense/meco_track_training.py`, which the accepted overall plan reserves exclusively to P5, and it does not pin an exact source base or exact fixture/test allowlist. A derivative needs substantive source-map, contract, and ownership redesign with a new review; that design is unresolved and outside this packet. No implementation was authorized from that plan.

The P0a approval resolves the protocol choices. It does not provide the P1 exact API, a P1 package allowlist, a P2 plan ACCEPT, G0, or runtime authorization. Exact seed/case bytes and opponent identities remain source-bound protocol outputs to freeze before new training or outcomes.

## Scope boundary

Until the package gates above are met, permitted work is planning and static inspection only. This packet did not edit runtime source, tests, build configuration, dashboard files, campaign files, checkpoints, or the user-modified HIP file. It did not run tests, builds, simulator evaluations, dashboard services, GPU/device operations, or training.
