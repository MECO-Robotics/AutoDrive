# P1 V4 schema and score-event source package plan

**Date:** 2026-10-26
**Status:** Revised Pass 1 successor candidate; previous exact-head review requested revision
**Planner:** `/root/p1_p2_planner` (planner only)
**Package:** P1 — V4 schema, typed score events, envelope and artifact API
**Implementation base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`

**Prior exact-head review:** REVISE at review commit
`1f201f611920948f79234426dd425a8664aeab03` for candidate commit
`376f3913976839a1c6043915c7b7854d449c921b`; the findings on complete G0
authorization, reviewed contract-anchor bridge, and two-WX9100 residency are
addressed in this revision.

## Relationship to accepted plans and authority

This is a finite successor/amendment to the accepted P1 plan at commit
`53686a254318484f9c6bc3496ecc1c461572c53a` (plan SHA-256
`21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076`, Pass 1
ACCEPT record commit `854d5286947074ed33f7ac059d67f2f9ccbf9257`, record SHA-256
`0680a57e5ff0427b20a0a5bd86c224085583b241b3f57bcb18bde69cab0f158f`). It
supersedes the staged-G0 amendment candidate at `606ba5adce04afd84bfb372260aebde8fd4fcec6`
only if this exact successor receives independent Pass 1 ACCEPT. The prior
accepted P1 plan's allowlist and acceptance boundary remain unchanged until
that review accepts this successor.

The accepted overall plan is commit
`8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, plan SHA-256
`57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`, with
Pass 1 ACCEPT record commit `b0b2962d299b1ecaab053b873ef2ae09e3c5f467` and
record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`.
P0a is accepted: candidate commit
`56183cefd2402342ce28f836155164fc850ebc60`, file SHA-256
`c8ace83e0e5a3fdef8d33ab090d3cbb9193e12342b015d72b07016acdb603726`;
maintainer decision record commit `7007607010636fe7451ad1b59eb1a900e0a4d7b1`,
file SHA-256 `d9394b07bc508021a4b10faf89b2d258a863183288d633442f796fc181592e64`;
exact-head Pass 3 ACCEPT record commit
`d5dab585ef1147f0d9997ece80db6e9a7ba95979`, record SHA-256
`25a2c38c502599a4da9c9448bdd3e6850b3ae7145d5977870e17b98304c280e6`.

The reconciled ABI artifact is pinned at commit
`03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, SHA-256
`294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; its
reconciled profile hash is
`5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`. The V4
proposal digest used in the 96-byte envelope is separately
`0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.

This document is a plan candidate only. It does not authorize source changes
until its exact-head Pass 1 ACCEPT and assignment of a distinct implementer
and reviewer. After those gates, P1 may author only its finite allowlist and
perform static review in a fresh worktree branched from the exact source base
above. Pre-G0 source/API review may accept or revise the exact candidate; it
does not authorize dependent integration, runtime use, or package completion.

## Fixed ABI contract

The reconciled profile fixes one track (`N=1`), six robot slots (`S=6`), 32
transition rows (`R=32`), 12 ticks per macro (`T=12`), and a 2,304-row event
table (`R×S×T`) of 59-byte records. The table occupies 135,936 bytes. The
payload is 314,469,051 bytes; the envelope is 96 bytes; the complete bundle
is 314,469,147 bytes. The canonical declaration, raw proposal digest,
little-endian packed field order, dimensions, capacities, code registries,
and byte totals are immutable in this package. Any required field/type/order,
capacity, or semantic-registry change returns to ABI reconciliation; this
plan authorizes no profile alteration.

The adopted event row fields are exactly:

| Field | Type | Bytes |
|---|---:|---:|
| `valid` | bool | 1 |
| `rollout_row_index` | uint8 | 1 |
| `episode_event_sequence` | uint16 | 2 |
| `tick` | int64 | 8 |
| `match_elapsed_seconds` | float64 | 8 |
| `world_slot` | int64 | 8 |
| `alliance` | uint8 | 1 |
| `robot_slot` | int64 | 8 |
| `piece_slot` | int64 | 8 |
| `piece_generation` | int64 | 8 |
| `event_type` | uint16 | 2 |
| `score_units` | uint16 | 2 |
| `source_site_code` | uint16 | 2 |

The Gamepiece/score owner adopted `event_type=1` for `fuel_score_awarded`,
`score_units=1` per accepted mutation, and site code 1 for the vectorized
score mutation and 2 for the per-robot score mutation. Every unused row is
zero except `valid=0`. Reserve the full table before score-state mutation;
validate/stage before committing score counters or respawn state; overflow or
invalid event data fails the whole uncommitted transition and emits no partial
table. P1 must not claim rollback of physics ticks that have already executed.

The ABI fixes all field names, types, widths, order, and codes. It does not
fully define the base source's physical-slot generation lifecycle or the
source clock conventions listed in the source-map section. This is a P1
implementation blocker for event production, not permission to choose values.

## Exact source map at the pinned base

The following map was obtained by static inspection of the exact base tree;
these are observed source facts, not V4-conforming behavior. SHA-256 values
pin the inspected source files.

| Role | Exact source site at base | Observed behavior / consumer boundary | File SHA-256 |
|---|---|---|---|
| Authoritative vectorized producer | `frc_defense/gamepiece_actions.py`, `update_fuel_actions`, lines 360–379; selected mask is formed at 363–367, aggregate event/counters are mutated at 368–379. | Selects at most one eligible physical piece per robot; current row has only a per-robot aggregate count. `piece_slot` is available as a tensor index during selection, but no event row is captured there. | `74c47eb546565f46712bd5328c50c71b07275b897cb1ef3fc9216b67d4ad260c` |
| Authoritative per-robot producer | Same function, lines 382–422; `chosen` is formed at 412–414 and aggregate counter mutations occur at 415–422. | Fallback robot loop also selects at most one piece per robot. It uses a host-static robot loop and static team ID; no typed event identity is emitted. | same as above |
| Producer dispatch | `frc_defense/_tensor_3v3_gamepieces.py`, `TensorThreeVsThreeGamepieceMixin._update_fuel`, lines 21–26; `_update_match` at 133–135. | Dispatches both source-site paths through `update_fuel_actions`. `tensor_3v3.step` advances `steps` and match clock before calling this method (lines 1340–1342). | `480042250dcf281142929c62abf8bac60c007c498fb6105d517c55d7f645746a` |
| Environment state/reset | `frc_defense/tensor_3v3.py`, score arrays and physical piece arrays at 305–322; full reset at 468–481; per-world reset at 534–546; step result at 1340–1410. | Maintains aggregate per-world/alliance scores and per-robot event counts; physical slots are reused on scoring/respawn. No physical `piece_generation` or episode event sequence tensor exists in this base. `world_slot` is 0 under the adopted `N=1` profile. | `e4925bfc100a487fbf734d0620cd7ab3b66ffc4e9027edb1cde78cbaca2f982e` |
| Current metric consumer | `frc_defense/meco_metric_ledger.py`, event map at 102–114. | Consumes `fuel_scored_event` as a per-focal-robot aggregate; it is diagnostic and not an event-ID ledger or score-award source. | `aa2a440c056a8a54978196aa4d0c6bf809ff807d78ae0d618a33ba8b26e3fb49` |
| Current evaluation/report consumer | `frc_defense/tensor_training_evaluation.py`, event totals at 260–269, aggregate ingestion at 478–489. | Accumulates aggregate `fuel_scored_event` counts and emits `total_simulated_score`; it cannot establish unique accepted-event IDs or whole-alliance source provenance. It is not the accepted P0a score ledger. | `72036f22011a03fdbcd5ac8f235608ee3a37716b18b8c449f1b289c2b0500483` |
| P5 trainer seam | `frc_defense/meco_track_training.py`, current action/score observation at 654–669 and trainer call sites identified in the accepted overall plan. | Shared trainer rollout/counterfactual/checkpoint code is exclusively P5-owned. P1 publishes a typed event/batch API and an immutable API pin; P5 alone connects that API to rollout storage, reward/return collection, and checkpoint manifests. | `da3cb88c320aa8a2945b6ad0c9d9b4dbd34ef843778eaabde8b0f3660a99b07b` |
| Accepted P0a ledger source | `frc_defense/meco_simulated_score_ledger.py` is **absent** from the pinned base tree. It exists at the later `main` evidence pin `d74b0a15e25ddbc6bacfbbb864674371629bb19e`, cited by the accepted P0a record. | The P0a definition is whole-alliance accepted awards summed by unique event ID, with red/blue totals and no per-robot causal claim. P1 must provide the source-bound event API that can support this definition; it must not silently copy later `main` implementation onto this base. | absent at base |

The source-site symbols are the adopted codes even though the owner response's
line numbers differ from this pinned base. The code anchors above show the
same two named branches. The score table is populated only from accepted
`chosen` physical-piece mutations, never from attempted score intent, event
counters alone, reward pulses, or curriculum metrics.

## Unresolved source semantics and stop rule

The accepted V4 table leaves **no field name, type, width, order, code, or
capacity unresolved**. The pinned implementation source still lacks fields
needed to populate adopted identities. Do not invent them in P1:

1. The physical piece slot has no source-backed `piece_generation` counter or
   initialization/increment/reset/exhaustion lifecycle. The plan must not
   assume generation 0 at initialization or choose a respawn increment rule.
2. No episode-local `episode_event_sequence:uint16` counter exists in the
   pinned source. A per-tick or loop-order serialization convention must not
   be guessed. P1.1 must establish a deterministic source order and exact
   sequence/reset rule that fits the adopted width and row-plus-episode
   identity.
3. `tick` zero/one-based convention and exact `match_elapsed_seconds`
   sampling convention are not stated by the ABI. The source advances
   `steps` and `match_elapsed` before scoring; P1.1 must document the accepted
   interpretation, not infer it from line order alone.
4. The accepted P0a authoritative ledger path is absent at the exact source
   base. P1.1 must reconcile the source map to the P0a-approved event
   aggregation. If the missing source path is a hard dependency, stop for a
   reviewed plan/source-base amendment instead of importing the later main
   branch wholesale.

P1.1's read-only source-map record may be authored first after this plan is
accepted. If it cannot resolve items 1–4 solely from the pinned source and
accepted contracts, stop before event producer/consumer code changes and
request a reviewed ABI/source-plan amendment. Do not create or use a new
generation, tick, elapsed-time, event-order, or ledger contract by agent
choice. Schema encoder/reader code may only encode the adopted fixed bytes; it
must not declare unresolved event semantics as normative.

## Package sequence and P5 handoff

1. **P1.1 source map:** document every score producer, physical mutation,
   reset, and consumer against the exact base. Resolve or explicitly stop on
   each item in the previous section. Publish its exact commit/hash as a
   prerequisite to event-path edits.
2. **P1.2 schema and artifact API:** implement the fixed V4 field/byte
   registry, serializer/reader, validation, digest domains, opaque artifact
   writer, and unused-row zeroing. Keep host code to static configuration and
   opaque persistence; dynamic row validation and digest remain device-side.
3. **P1.3 score producer:** only after P1.1 resolves the identity/clock/order
   items, stage events at both accepted mutation sites and expose the typed
   event rows/status at the simulator boundary. Preserve existing score,
   reward, RNG, and physics rules. Fail the uncommitted transition on
   overflow/malformed identity; never host-repair or partially publish.
4. **P1.4 artifacts/source binding:** bind source/config/checkpoint/profile
   identity and the fixed payload to the strict v2 envelope. P1 provides an
   immutable typed-batch API and its exact source/API pin to P5.
5. **P5-only trainer handoff:** P5 owns every edit to
   `frc_defense/meco_track_training.py`, trainer rollout/counterfactual
   consumers, PPO/RNG/checkpoint state, and associated trainer integration
   tests. P1 must not modify those paths. P5 may begin dependent source
   authoring only after an independent pre-G0 SOURCE/API ACCEPT pins the exact
   P1 API head; it may not integrate or runtime-use that API before G0.

P1.3 cannot start unless P1.1 closes the source semantics without changing the
accepted ABI. A profile-byte change or required new event field returns to
ABI reconciliation and requires a new exact profile hash.

## Finite Pass 2 allowlist

Pass 2 may edit only these paths in a new ZFS worktree based on the exact base
above. New paths are planned outputs and are not claimed to exist at the base.

| Path | Ownership |
|---|---|
| `docs/implementation-provenance/meco-v4-p1-score-source-map-20261026.md` | P1.1 source map and field-level resolution/stop record. |
| `frc_defense/meco_transition_v4.py` | Fixed V4 transition/event field registry, exact packing/reader, validation, digest interface, and opaque artifact construction. |
| `frc_defense/meco_score_events_v4.py` | Fixed event table allocation/staging, event row validation, ordering interface, failure state, and event-derived alliance projection. |
| `frc_defense/gamepiece_actions.py` | Narrow P1 hunks at the two accepted scoring mutation sites only; gated on P1.1. |
| `frc_defense/_tensor_3v3_gamepieces.py` | P1 event-buffer initialization/reset lifecycle only, gated on P1.1. |
| `frc_defense/tensor_3v3.py` | P1 event-buffer allocation/reset and typed event output seam only; do not edit general environment/P2 residency logic. |
| `tests/test_meco_transition_v4.py` | New exact bytes, field layout, length, digest, and reader fixtures. |
| `tests/test_meco_score_events_v4.py` | New deterministic identity/order/reset/capacity/failure fixtures, after P1.1 closure. |
| `tests/test_meco_transition_v4_simulator.py` | New end-to-end scenario fixture authored only; execution gated by G0 and AGENTS.md requirements. |

No other path is authorized. In particular, P1 may not edit
`frc_defense/meco_track_training.py`, `frc_defense/meco_metric_ledger.py`,
`frc_defense/tensor_training_evaluation.py`, dashboards, build configuration,
native HIP files, or campaign/checkpoint/output data. If source discovery
requires an unlisted path, stop and obtain a Pass 1 allowlist amendment. The
user-modified HIP file remains outside this task.

## Acceptance and runtime gates

### Pre-G0 source/API acceptance

An independent reviewer different from planner and implementer checks the
exact pushed head, path allowlist, fixed ABI declaration, source-map evidence,
P1.1 stop/resolution record, source/API boundary to P5, and static diff. It
records SOURCE/API ACCEPT or REVISE for that exact head. It does not run
imports/project code, tests, builds, kernels, profilers, simulation,
dashboard, or devices. SOURCE/API ACCEPT pins a candidate for dependent source
authoring only; it is not package-final acceptance or integration authority.

Before G0, source/test-file authoring and static review are the only code work
allowed. Do not run a test, compile/build, import project code, simulator,
calibration, evaluation, profiler, dashboard, GPU/device access, training,
checkpoint mutation, integration, merge, or promotion. Do not inspect, stop,
signal, reconfigure, or otherwise touch the active campaign, its outputs,
workers, checkpoints, dashboard, devices, or leases.

### Post-G0 package acceptance

No validation, runtime/device operation, dashboard operation, or integration
may begin until all G0 evidence is recorded: (1) the user has manually stopped
campaign PID `2750253`; (2) both assigned WX9100 leases are independently
confirmed released by their lease owners; and (3) the user or repository
maintainer explicitly authorizes runtime/device/integration work. Do not infer
any condition from elapsed time or telemetry.

After those G0 conditions, create a docs-only contract-anchor bridge that adds
the exact accepted reconciliation artifact bytes at their pinned path. Verify
the artifact SHA-256 remains
`294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`, the
canonical profile hash is unchanged, and the implementation source tree is
identical to `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`. An independent
reviewer must accept that exact bridge head before it becomes the integration
and validation base. No P1 validation or integration may start before this
reviewed bridge.

All P1 algorithmic score/event/digest computation must execute on each of the
two assigned AMD WX9100 devices under recorded UUID/runtime identity. CPU
numerical computation, CPU score/event validation or digesting, P600
substitution, and silent device migration are prohibited. CPU use is limited
to process supervision, configuration, and opaque filesystem/network I/O.
If either WX9100 or a required HIP kernel path is unavailable, fail closed and
stop; do not fall back. Retain per-process device identity, kernel/device
placement, memory ownership, and transfer/synchronization evidence for both
devices.

After the reviewed bridge, run the exact accepted tests plus a reproducible
end-to-end simulator evaluation covering both score branches, event identity,
and record/replay reconstruction. Restart the dashboard services from the
current integration checkout before replay review; verify backend/frontend
provenance, service health, and endpoints; generate a fresh replay and verify
its source. Record tests, scenario, seed, replay ID, source and event hashes,
both device UUIDs/runtime identities and residency evidence, dashboard
currentness, and outcomes. The final independent exact-head Pass 3 review is
required before integration. No runtime, simulator, GPU, or dashboard evidence
exists as part of this plan.

## Reviewer / implementer roles

- Pass 1 planner: `/root/p1_p2_planner`.
- Pass 2 implementer: unassigned.
- Pre-G0 source/API reviewer: unassigned and distinct from planner and
  implementer.
- Post-G0 final Pass 3 reviewer: unassigned and distinct from author.

This candidate is planning-only and does not claim source conformance, score
correctness, GPU residency, simulator success, current dashboard state,
training progress, or qualification. No source, test, build, simulator,
dashboard, device, campaign, checkpoint, or output operation was performed
to prepare it.
