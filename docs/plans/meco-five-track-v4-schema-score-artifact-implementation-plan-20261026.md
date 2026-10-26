# V4 schema, typed score-event, and artifact implementation plan

**Date:** 2026-10-26  
**Status:** Planner-authored candidate; pending independent Pass 1 review  
**Planner/task identity:** `/root/plan_c_owner`  
**Implementation base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`  
**Accepted reconciliation:** `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`  
**Reconciliation file SHA-256:** `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`  
**Reconciliation acceptance:** `5cb1a2d525168ff0443089cf329e66c6ecb917b8`  
**Canonical reconciled profile hash:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`  
**V4 proposal commit / profile digest:** `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b` / `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`

## Purpose and authority

This plan defines a bounded P1 implementation package for the reconciled V4
transition schema, score-event producer/consumer, and fixed artifact envelope.
It does not implement five policy architectures, PPO/GAE, curriculum, or the
full GPU residency conversion. Those remain separately scoped work. The
package is about producing, validating, and persisting the exact reconciled
transition bytes, with typed score-event provenance from the simulator's
actual scoring mutation.

This document is planning only. It is not approval to edit source, run tests,
build, use devices, run a simulator, restart a dashboard, publish ABI bytes, or
start training. It makes no claim that the base source conforms. A distinct
implementer and independent reviewer must be assigned after Pass 1; they may
not be the same person or agent. Each review must pin an exact pushed commit
and file hash. Implementation remains gated on the external G0 release and
the accepted P0 score-source/rules artifact described below.

## Contract pins

The accepted reconciliation binds the exact V4 proposal, owner decisions,
profile, and byte totals. The implementation target is its canonical profile
declaration, hash `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`.
The V4 `profile_digest` is separately the raw proposal-file digest
`0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
Do not substitute the reconciled-profile hash for the V4 profile digest.

The profile declares `N=1`, `S=6`, `R=32`, `T=12`, a fixed score-event table
of 2,304 records × 59 bytes (135,936 bytes), a 314,469,051-byte payload, a
96-byte header, and a 314,469,147-byte complete bundle. Its score-event row is
the accepted 13-field little-endian packed layout: `valid:bool`,
`rollout_row_index:uint8`, `episode_event_sequence:uint16`, `tick:int64`,
`match_elapsed_seconds:float64`, `world_slot:int64`, `alliance:uint8`,
`robot_slot:int64`, `piece_slot:int64`, `piece_generation:int64`,
`event_type:uint16`, `score_units:uint16`, and `source_site_code:uint16`.
The accepted Gamepiece/score row assigns `event_type=1` to
`fuel_score_awarded`; source-site codes 1 and 2 distinguish the vectorized
score mutation and per-robot score mutation. The full capacity is serialized;
an unused row has `valid=0` and all other fields zero.

The envelope uses the accepted version-first 96-byte header, exact-length
reader, raw-proposal `profile_digest`, and separate `source_digest`,
`config_digest`, `checkpoint_digest`, and `payload_digest` domains. The host
may persist opaque bytes; live validation, event handling, and digest
computation remain on the assigned GPU path. The fixed layout and fail-closed
rules are not adjustable inside this package. A required byte or profile
change stops work and returns to reconciliation with a new profile hash.

## Hard prerequisites and stop conditions

### G0: campaign and resource release

The eventual implementer must verify the accepted implementation plan's G0
conditions before source work: the active campaign has been stopped by the
user, and the assigned GPU leases have been independently confirmed released.
No task in this plan may inspect, stop, signal, reconfigure, evaluate through,
or otherwise touch an active campaign or its dashboard, workers, checkpoints,
output paths, leases, or devices. Do not infer G0 from elapsed time or from
absence of a process observed in this planning task.

### P0: source/rules-bound score contract

The P0 score-source/rules artifact is an upstream dependency and is not
created, guessed, or approved by this plan. Before implementation, it must be
immutable, source-bound, independently reviewed, and carry the required human
decision/approval provenance. Its digest and exact URI/path must be pinned in
the implementation ledger. It must define, at minimum:

- The exact current score-producing mutation and accepted event boundary for
  the user-selected `simulated_match_score` estimand: accepted active-HUB FUEL
  awards only, separate from official match points.
- The meaning of one score event and `score_units`; any additional score event
  types, sources, or units are prohibited unless the V4 contract is amended.
- The physical identity domain: simulator `piece_slot` and its generation
  lifecycle at scoring, respawn, reuse, full/partial reset, and environment
  reconstruction. Observation-track handles are a different identity domain.
- `tick` meaning, `match_elapsed_seconds` source, alliance/robot-slot mapping,
  event order, episode sequence reset, reset-generation binding, and overflow
  behavior. The `uint16` sequence bound must be proven from authoritative source
  or a new profile decision; the cited later recorder limit is not silently
  assumed to exist in this base checkout.
- The simulator's authoritative producer and every metric/report consumer,
  including where event-derived score is accumulated and where it is merely
  presented. If the artifact disagrees with the accepted event codes or
  capacity, stop and request reconciliation; do not change codes locally.

No Phase 0 artifact hash, human approval, score rule, or approved seed
partition is currently supplied by this task. Until P0 is complete, later
packages may be planned but may not start implementation or qualification.

## Exact base inspection and source-discovery gate

The base commit is immutable for this package. Initial read-only inspection
locates the visible scoring mutation in
[`gamepiece_actions.py`](../../frc_defense/gamepiece_actions.py):
`update_fuel_actions` clears per-robot event counters at lines 62–66; the
vectorized score path selects eligible physical piece indices and updates
`fuel_scored_event` / `fuel_score_count` at lines 331–381; the alternate
per-robot path does so at lines 382–422; and selected pieces are released and
respawned at lines 424–453. The 3v3 mixin dispatches to this function at
[`_tensor_3v3_gamepieces.py`](../../frc_defense/_tensor_3v3_gamepieces.py)
lines 21–26. The environment allocates score counters and physical piece
arrays in [`tensor_3v3.py`](../../frc_defense/tensor_3v3.py) lines 305–322,
resets counters and fuel at lines 468–481 and 534–545, and returns aggregate
score-event tensors from a step at lines 1401–1407. The metric ledger samples
per-step event tensors in [`meco_metric_ledger.py`](../../frc_defense/meco_metric_ledger.py)
lines 102–114; it is not a typed V4 event table.

This is only an initial map of inspected files, not proof those are the only
producers or consumers. The accepted Gamepiece/score response cites a
`meco_simulated_score_ledger.py` path and line ranges that are not present at
that path in the pinned base tree, and its cited source-site line numbers do
not match the score-mutation lines in this base. Do not infer that the named
module exists, create a replacement for it, or treat the response citation as
evidence for this checkout. P1.1 must search the pinned tree, imports, package
metadata, generated/native dependencies, and relevant history, and return an
exact source map. If that does not locate the authoritative episode event
source/consumer or resolve the cited source mismatch, stop for a P0/plan
amendment before implementation. Do not expand the allowlist to an unreviewed
path.

## Finite package sequence

The work is sequential. Every package produces a pushed commit and exact-head
review before its dependent package starts. A REVISE returns only that package
to its owner; a changed schema, profile, event code, source contract, or byte
total stops the sequence pending reconciliation.

### P1.1 — Read-only source map and interface decision record

**Dependency:** G0 and accepted P0 artifact. No source edits or runtime work.

Trace each producer, mutation, reset, and consumer from the base tree. Record
the exact path/line, device/dtype/shape, source event, physical piece identity,
source-site branch, reset boundary, and current transfer/persistence path.
Resolve all still-open row-level meanings that do not change V4 bytes,
including deterministic valid-row ordering and how the 2,304 reserved slots
are filled across a 32-row batch. The record must distinguish accepted owner
decisions from observed source behavior. If the chosen interpretation changes
any field, type, identity domain, code, capacity, offset, or byte, stop for
reconciliation.

**Output:** an implementation-ledger entry containing the P0 artifact
identifier/hash, source map, V4 profile hash, exact current base, per-file
owner, producer/consumer interface, and an explicit resolve/hold decision for
each open item above. The source map is planning evidence, not a substitute for
P0.

### P1.2 — V4 typed schema, encoding, digests, and strict reader

**Dependencies:** accepted P1.1 decision record and immutable profile pins.

Implement the typed field declarations and exact row/batch layout, versioned
96-byte header, checked offset arithmetic, little-endian field packing,
device-side validity checks and digest computation, and an exact-length
version-first reader. Keep the 59-byte record packed without implicit native
struct alignment. The reader rejects unsupported versions before parsing
payload fields; mismatched profile, lengths, codes, digests, reserved bits,
invalid valid-fields, or overflow reject the whole batch before any action,
optimizer, RNG, or checkpoint state changes. Unused event records are fully
zeroed apart from `valid=0`. Persist the exact opaque bundle; no host-side
repair, hash, decode, aggregation, or selection of live payload values.

**Output/interface:** one GPU-resident V4 batch object accepted by the trainer
and artifact writer, and a strict reader that returns validated typed views
only after version, length, digest, profile, registry, and mask checks. The
artifact writer returns the exact immutable bundle bytes and manifest identity;
it may stage a file write but may not publish a partial bundle.

### P1.3 — Physical score identity and device-native event staging

**Dependencies:** P1.1 source map, accepted P0 score/rules artifact, P1.2
schema interfaces.

At both actual score mutation sites, capture the physical selected
`piece_slot`, its adopted `piece_generation`, alliance/robot, score time/tick,
rollout-row and episode-sequence identity, event type, score units, and
accepted source-site code before score counters, piece ownership, or respawn
state commit. Implement generation initialization/increment/reset semantics
only where the source map proves the correct physical slot lifecycle. Keep
the existing gameplay rule and selected-piece behavior unchanged.

Reserve the complete fixed 2,304-event staging table before the first
simulator mutation for the batch. Stage event rows and all dependent score
mutation until the row validates. If capture, capacity, generation, sequence,
or event validation fails, reject the complete uncommitted transition/update;
do not truncate, host-repair, increment only an aggregate counter, or publish
a partial event/artifact. If current environment stepping cannot stage the
affected mutation, stop and return with a revised mutation boundary plan; do
not claim rollback of already executed physics ticks. Runtime capture and
validation must remain on the assigned GPU, with no Python scalar extraction
or CPU-list path for live event values.

**Output/interface:** the simulator produces only typed staged V4 score rows
plus explicit failure status. The consumer receives the same committed event
rows used by the aggregate score update; it must not reconstruct physical
piece identity from aggregate `fuel_scored_event` counters.

### P1.4 — Consumer, bundle assembly, and checkpoint/manifest binding

**Dependencies:** accepted P1.2 and P1.3 interfaces.

Connect the event table to the reconciled transition-row assembly and to the
source/rules-bound score consumer identified in P1.1. Prove that event count
and event-derived `simulated_match_score` agree with the accepted P0 rule,
while route/risk shaping and official-score representations remain separate.
Bind the profile, source, configuration, checkpoint, track, schema/envelope,
protocol, and payload digest fields exactly as adopted. Use a versioned V4
manifest/reader contract; the base's schema-version-1 checkpoint or any JSON
metric ledger is not a V4 substitute. Preserve atomic publication and
all-or-nothing state handling. Do not change PPO reward, GAE, terminal,
truncation, optimizer, curriculum, or model behavior in this package.

**Output/interface:** a validated transition batch and immutable matching
manifest, linked by exact source/config/checkpoint/profile/protocol identity
and byte digests, consumable by the separately owned trainer/evaluation
integration at its agreed boundary.

### P1.5 — Exact-head review, seeded simulator and dashboard evidence

**Dependencies:** P1.1–P1.4 accepted; G0/P0 remain satisfied; current checkout
integrated from the exact reviewed commits.

Run only the package-authorized schema/event tests and representative
end-to-end simulator evaluations. Use the P0-frozen scenario/seed IDs; include
score-producing scenarios exercising each supported score-capture branch
without changing the event rule. Inspect the fresh simulator record/replay:
valid events must tie to the physical piece incarnation and actual accepted
score mutation, unused rows must remain zero, aggregate output must match the
P0 estimand, and failure injection must leave no partial score/event/artifact
commit. Execute numerical simulation and event capture on each assigned WX9100
under its recorded UUID/runtime. CPU is limited to process supervision,
configuration, and opaque filesystem/network I/O; CPU simulation, score
calculation, digesting, metric calculation, silent migration, and P600 fallback
are failures.

Before generating or reviewing a dashboard replay, restart the serving
dashboard service(s) from the current integration checkout. Verify backend
and frontend asset provenance, active service state, and successful response
from each relevant endpoint. Generate a new replay after restart and check its
source/replay provenance. Mark old-code replay artifacts outdated; do not
reuse them as evidence. Retain scenario ID, exact seed, replay ID, simulator
record, device UUID/runtime, service/process identity, asset identity, and
endpoint results in the review evidence.

## Exact implementation allowlist

Pass 2 may modify only these paths. No other source, schema, test, dashboard,
campaign, checkpoint, or data path is authorized by this plan.

| Path | P1 authority |
|---|---|
| `frc_defense/meco_transition_v4.py` | New V4 field registry, fixed-layout encoder/reader, validation, and digest interfaces. |
| `frc_defense/meco_score_events_v4.py` | New device-resident score-event staging/validation and batch assembly helpers. |
| `frc_defense/gamepiece_actions.py` | Capture and stage only at proven score-mutation sites; preserve existing scoring behavior. |
| `frc_defense/_tensor_3v3_gamepieces.py` | Physical score-generation lifecycle only after source discovery resolves initialization/reset ownership. |
| `frc_defense/tensor_3v3.py` | V4-scoped event/generation buffers, reset integration, and typed transition output seam only. |
| `frc_defense/meco_track_training.py` | V4 batch packing and manifest boundary only; no PPO/model/reward/GAE/curriculum edits. |
| `tests/test_meco_transition_v4.py` | New schema, exact-byte, digest, length, and reader fixtures. |
| `tests/test_meco_score_events_v4.py` | New event ordering, identity/generation, reset, capacity, and atomic-failure fixtures. |
| `tests/test_meco_transition_v4_simulator.py` | New seeded end-to-end score mutation/replay fixture. |
| `tests/test_fuel_physics_integration.py` | Existing integration assertions only if required for source-site behavior coverage. |
| `tests/test_tensor_3v3_physical_constraints.py` | Existing scoring regression assertions only if required for source-site behavior coverage. |

`frc_defense/meco_simulated_score_ledger.py` is intentionally not on the
allowlist: the path is not present in the pinned base checkout. P1.1 must
establish whether an authoritative source is present elsewhere or whether P0
needs a separately approved source decision. If a discovered source requires
editing an unlisted file, stop and obtain a Pass 1 plan amendment before
touching it. No native HIP source is currently allowlisted; add one only by a
reviewed plan amendment backed by device profiling and an exact interface
proposal. New modules in this table are planned outputs, not claims that
similarly named modules already exist.

## Test fixtures and acceptance evidence

Pass 2 must add deterministic fixtures for:

1. Exact 59-byte event field order/types/endianness, zero-filled invalid rows,
   all 2,304 reserved slots, and independent arithmetic for 135,936-byte event
   table, 314,469,051-byte payload, 96-byte header, and 314,469,147-byte
   bundle. Include signed zero and finite-mask behavior where present in the
   V4 row layout.
2. Version-first header rejection, unknown schema/profile, reserved-byte and
   code failures, short/oversized/trailing bundles, checked-length overflow,
   payload/source/config/checkpoint digest mismatch, and no partial reader or
   artifact publication.
3. Vectorized and per-robot score-site capture; one event per accepted physical
   score mutation; deterministic source-site and slot ordering; no duplicate
   event for one mutation; piece slot/generation preservation before respawn;
   reset/sequence boundaries; invalid/stale physical identities; sequence and
   capacity exhaustion. Any bound not proven by P0 remains a blocked fixture,
   not a fabricated expected value.
4. Fault injection before and during staging, before score-state commit, and
   before artifact rename/publication. Assert the specified no-op state for
   score counters, piece ownership/respawn, event table, artifact, optimizer,
   RNG, and checkpoint as applicable. If a physical tick already mutated and
   cannot be rolled back, the implementation must demonstrate it stages the
   package-owned score mutation before that boundary or halt for redesign.
5. A byte-identical GPU-produced bundle and digest against fixed independent
   fixtures. Any host reference encoder is test-only and may not enter the
   live producer, digest, reader, or decision path.

**Pass 2 ACCEPT evidence must include:** exact P0 artifact URI/hash and approval
provenance; exact source map; accepted P1.1 decision record; exact base and
profile hashes; per-package pushed commits and file allowlist; all fixture
results; GPU UUID/runtime evidence for both WX9100s; no-CPU transfer/sync trace
for the event-to-artifact path; and exact artifact/manifest hashes. Pass 3 must
re-open the exact pushed head/blob, review all allowlist changes and fixture
results, and return a persisted ACCEPT/REVISE. No mutable branch-tip review is
accepted.

For every test that checks scoring or other simulated behavior, evidence must
also contain the representative seeded simulator evaluation and inspected
record/replay. Report each test name, scenario, exact seed, replay identifier,
result, device UUID/runtime, and whether dashboard backend/frontend freshness
and endpoints were verified. If any required simulator or dashboard gate
cannot run, mark validation incomplete; a unit/regression pass is not an
end-to-end pass.

## Risks and explicit open decisions

- **P0 score/rules dependency:** No source/rules-bound Phase 0 artifact,
  approval provenance, or frozen seeds were supplied. The score estimand,
  source boundary, tick/episode sequence semantics, max sequence, and reset
  behavior remain blocked on that artifact.
- **Source mismatch:** The accepted owner response describes a named recorder
  module and line references that do not match the exact pinned base. P1.1
  must reconcile this against the real base tree and history before capture is
  designed. Do not manufacture the module or treat response citations as
  current source evidence.
- **Physical generation:** The base has piece slots and aggregate event
  counters, but this inspection did not find V4 `piece_generation` or
  `episode_event_sequence` storage. Generation allocation, increment timing,
  exhaustion and reset semantics need source-backed resolution; observed-track
  generation is not a substitute.
- **Mutation atomicity:** Current scoring mutates counters and later releases/
  respawns pieces in the same update function. Event validation must precede
  these mutations or use a reviewed staging boundary. There is no permitted
  best-effort post-mutation recorder path.
- **Byte and memory scale:** A full payload is about 300 MiB before staging,
  digest workspace, or checkpoint buffers. Prove peak memory and device
  placement on the assigned WX9100s. If capacity cannot fit without copies or
  changing the profile, stop for reconciliation; do not silently shrink,
  truncate, or use CPU.
- **Concurrency:** `gamepiece_actions.py`, `tensor_3v3.py`, and
  `meco_track_training.py` have other owners. The implementation coordinator
  must reserve these paths to this package for the specified hunks or defer
  conflicting edits. PPO, curriculum, dashboard, and campaign work remain
  outside the allowlist.

## Completion and non-goals

P1 is complete only when the exact reconciled profile is implemented for
schema, score-event producer and consumer, and bundle/manifest; the P0 source
contract is independently accepted and pinned; exact-byte, digest, failure,
and source-identity fixtures pass; required seeded simulator records/replays
show the changed scoring behavior; both assigned GPUs are verified; and the
current dashboard backend/frontend pass the freshness and endpoint gate.
The P1 package does not qualify A1/A2/A3/C1/C2, authorize continuous training,
publish V4 as a repository-wide normative ABI, or replace subsequent PPO,
curriculum, model, GPU-residency, and Stage 1 plans.

No source change, test, build, simulator, GPU, dashboard, checkpoint,
campaign, or training action was performed while preparing this plan.
