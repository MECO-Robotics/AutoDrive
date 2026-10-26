# P1/P2 source-authoring readiness packet

**Date:** 2026-10-26  
**Purpose:** Pin the first permitted source-only slice after P0a acceptance and identify the P1/P2 gates. This is planning documentation only; it does not authorize source edits, validation, or integration.

## Decision

No P1 or P2 source-only slice is authorized yet. P0a is accepted, but the accepted overall plan requires each package to have its own finite Pass 1-accepted package plan, named distinct implementer and reviewer, an exact base, and accepted dependency heads. An accepted P1 package plan and finite allowlist already exist, but that exact plan requires G0 before P1 source discovery/authoring. Its later staged-gate amendment is pending review and cannot be treated as pre-G0 authorization. There is no P1 source candidate with an independent SOURCE/API ACCEPT record.

The first source-only slice would be **P1.1's pinned score-event source map and V4 schema/API authoring** on base `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`, but it may start only if the new successor plan candidate receives exact-head Pass 1 ACCEPT and distinct Pass 2/Pass 3 roles are assigned. That candidate includes a finite allowlist and makes P1.1 source mapping/schema work pre-G0 while gating score-site producer edits on resolution of event identity/clock semantics. The candidate remains pending review; it grants no source authority yet.

G0 does not gate P1/P2 source authoring or static review. It does gate builds, test execution, numerical/runtime work, simulator/profiler, device/GPU access, dashboard work, training, checkpoint mutation, integration, merge, and promotion. The active campaign and device leases remain outside this packet's authority.

## Package gates

| Package | Source-only status | Exact start gate | Interface dependency / blocked work |
|---|---|---|---|
| P1 — V4 schema, score events, envelope and artifacts | **Not currently authorized pre-G0.** The accepted P1 plan has a finite allowlist but its source gate requires G0; the staged amendment is pending; the new successor candidate is pending Pass 1 review. | Exact-head Pass 1 ACCEPT of `meco-v4-p1-score-event-source-package-plan-20261026.md` and assignment of distinct implementer/reviewer. Then P1 may author P1.1 source-map/schema/API files on `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`. Producer edits still wait for P1.1 semantic closure; if the source/contracts cannot resolve it, request an owner/ABI amendment. | P1 must produce an exact API head with independent pre-G0 SOURCE/API ACCEPT before P2 source authoring. P5 alone owns `meco_track_training.py` trainer integration. No integration/runtime use before G0. |
| P2 — GPU-resident environment, observation, candidate and route core | **Source authoring not yet unblocked.** Corrected P2 plan has Pass 1 ACCEPT; its exact P1 API dependency is still absent. | Corrected P2 plan commit `09cf081a4b47b66a9b3fcecf3c3748eda5722465` (file SHA-256 `eb034463a5c4d0c655fe8e16d7636247ab899ec56bca1c4655122f725c5041d1`) received Pass 1 ACCEPT in review commit `de260bde20b30368d84531767cc0e8d3bf072b35` (record SHA-256 `fe66a8eccb71d7be58d6cc37a20e98f32132ce36f03fa0f09ec0aa4ec5370d18`). P2 may start only after the exact P1 candidate API/source head receives independent pre-G0 SOURCE/API ACCEPT and its own roles are assigned. | P1 source head is an API reference only, not an integration base. Neither package may integrate or runtime-use it before G0. |

## Exact evidence pins

- Accepted overall plan: commit `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`; plan SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; Pass 1 ACCEPT record commit `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`.
- Accepted P0a candidate: commit `56183cefd2402342ce28f836155164fc850ebc60`; file SHA-256 `c8ace83e0e5a3fdef8d33ab090d3cbb9193e12342b015d72b07016acdb603726`.
- P0a maintainer decision record (user reply “Approve all”): commit `7007607010636fe7451ad1b59eb1a900e0a4d7b1`; file SHA-256 `d9394b07bc508021a4b10faf89b2d258a863183288d633442f796fc181592e64`.
- P0a exact-head Pass 3 ACCEPT record: reviewer commit `d5dab585ef1147f0d9997ece80db6e9a7ba95979`; record SHA-256 `25a2c38c502599a4da9c9448bdd3e6850b3ae7145d5977870e17b98304c280e6`.
- Approved implementation source base: `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.
- Accepted P1 package plan: commit `53686a254318484f9c6bc3496ecc1c461572c53a`, plan SHA-256 `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076`; Pass 1 ACCEPT record commit `854d5286947074ed33f7ac059d67f2f9ccbf9257`, record SHA-256 `0680a57e5ff0427b20a0a5bd86c224085583b241b3f57bcb18bde69cab0f158f`. It has an exact implementation allowlist but its accepted source gate requires G0 before P1.1/source authoring.
- Staged P1 G0/source-authoring amendment candidate: commit `606ba5adce04afd84bfb372260aebde8fd4fcec6`, file SHA-256 `7dc5be35d3826a6047653021e97a941d6872a73a6997ba768e95753f693b0653`; status remains pending independent exact-head Pass 1 review. It is superseded only if the new successor candidate is accepted.
- New P1 successor candidate: branch `codex/p1-score-event-source-package-plan-20261026`, commit `376f3913976839a1c6043915c7b7854d449c921b`, file SHA-256 `b0c25979f1bfe8ab43f9d1ff784c40fe6e794f71df2b42ec363b53e5091d6571`; Pass 1 review is pending. It pins the exact producer/consumer map, P5 handoff, finite allowlist, and explicit stop semantics.
- Accepted corrected P2 plan: commit `09cf081a4b47b66a9b3fcecf3c3748eda5722465`, file SHA-256 `eb034463a5c4d0c655fe8e16d7636247ab899ec56bca1c4655122f725c5041d1`; Pass 1 ACCEPT record commit `de260bde20b30368d84531767cc0e8d3bf072b35`, record SHA-256 `fe66a8eccb71d7be58d6cc37a20e98f32132ce36f03fa0f09ec0aa4ec5370d18`.
- Reconciled profile V4 envelope digest: `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`; reconciled artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`.

## Existing score-event plan assessment

The existing `gpu-resident-score-event-buffer` plan (`codex/gpu-score-event-buffer-plan-20261026`, head `47ff1037dc5bf0035b5a1d4523f5bddae240351e`, file `docs/plans/gpu-resident-score-event-buffer-20261026.md`, SHA-256 `3095cbee304f21d1d84ae91c1152722f955df47fba58f2399d3e6302d518eff1`) is useful prior design evidence but cannot become the P1 package plan through a status/pin amendment alone. It targets the legacy episode recorder cap of 32,768 accepted events and a passive report drain. The reconciled P1 ABI instead requires a distinct fixed 2,304 × 59-byte batch table, full-capacity staging before mutation, and all-or-nothing transition failure. The existing plan also requires editing `frc_defense/meco_track_training.py`, which the accepted overall plan reserves exclusively to P5, and it does not pin an exact source base or exact fixture/test allowlist. A derivative needs substantive source-map, contract, and ownership redesign with a new review; that design is unresolved and outside this packet. No implementation was authorized from that plan.

The P0a approval resolves the protocol choices. It does not provide the P1 exact API, a P1 package allowlist, a P2 plan ACCEPT, G0, or runtime authorization. Exact seed/case bytes and opponent identities remain source-bound protocol outputs to freeze before new training or outcomes.

## Scope boundary

Until the package gates above are met, permitted work is planning and static inspection only. This packet did not edit runtime source, tests, build configuration, dashboard files, campaign files, checkpoints, or the user-modified HIP file. It did not run tests, builds, simulator evaluations, dashboard services, GPU/device operations, or training.
