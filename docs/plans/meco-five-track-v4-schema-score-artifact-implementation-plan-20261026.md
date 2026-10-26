# V4 schema, typed score-event, and artifact implementation plan

**Date:** 2026-10-26  
**Status:** Staged G0 execution-gate amendment; pending independent exact-head Pass 1 review

**Plan author:** `/root/plan_c_owner`; **amendment author:** `/root/abi_freeze_planner`

**Amends accepted P1 plan:** commit `53686a254318484f9c6bc3496ecc1c461572c53a`, plan SHA-256 `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076`
**Implementation base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`  
**Accepted reconciliation:** `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`  
**Reconciliation file SHA-256:** `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`  
**Reconciliation acceptance:** `5cb1a2d525168ff0443089cf329e66c6ecb917b8`  
**Overall-plan P0a/P0b amendment:** commit `6325920252773cfa05d63167ba3c29a90debdb01`, file SHA-256 `9a12708980529cb03810ad40f43dbfe1e29461485a4f50fd1a7b06c7202dec5a`; Pass 1 ACCEPT record commit `be845d7fb2d2ece861dcd158ac8b8dd89e3dc59d`, file `docs/plans/reviews/meco-five-track-v4-gpu-phase0-amendment-6325920.md`, SHA-256 `5a6e94f978c8a7e18d9623897ceba6f32f5b5eadb0aba6605f21c9d1e941a886`
**Scoped P0 package plan:** commit `13dbd7f8f953995850e811202f7e9c5264c49dca`, file `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md`, SHA-256 `09ec722d64d01d99b60cc9ed52e47ce68790e548f66107cc1a19dd9a7a588adf`, Git blob `30df1bbed8657b08c1553db7487e59b8a1a0ad86`; Pass 1 ACCEPT record commit `037f0f9386cc0efadb1fb54efe5f38f097926d0e`, file `docs/plans/reviews/meco-v4-p0-phase0-scoped-package-plan-20261026.md`, SHA-256 `1ee70a6e9d24dc5201da5eec446820167cb8c4d4436fbbe013eb16de66b0280d`
**Prior P1 Pass 1 reviews:** REVISE at commit `c92480036ffb41161a444184f0f488059d281452` (P0 sequencing) and `a16cdd97742c20e18d4aec51e5c76077fd9aa46f` (P5 trainer ownership); superseded for the corrected base plan.
**Prior P1 Pass 1 ACCEPT:** reviewed plan commit `53686a254318484f9c6bc3496ecc1c461572c53a`, plan SHA-256 `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076`; review record commit `854d5286947074ed33f7ac059d67f2f9ccbf9257`, record SHA-256 `0680a57e5ff0427b20a0a5bd86c224085583b241b3f57bcb18bde69cab0f158f`. That ACCEPT does not cover this staged-gate amendment; this exact revision requires fresh Pass 1.
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

This document is planning only. It does not itself authorize source edits,
tests, builds, device use, simulator runs, dashboard restarts, ABI publication,
or training. It makes no claim that the base source conforms. A distinct
implementer and independent reviewer must be assigned after Pass 1; they may
not be the same person or agent. Each review must pin an exact pushed commit
and file hash. This plan follows the accepted overall plan pinned above. The
scoped P0 package plan must receive Pass 1 ACCEPT before P0a work begins; this
P1 plan and the overall plan must receive Pass 1 ACCEPT before P1 work begins.
After accepted P0a, the accepted overall plan, and an exact-head Pass 1 ACCEPT
of this plan revision, isolated worktree source authoring and static
source/interface review may proceed while
supervisor PID 2750253 remains live. G0 gates every test, build, import that
executes project code, runtime, device/GPU, simulator, dashboard, integration,
merge, promotion, and final package acceptance. P0b follows the reviewed
P1–P6 source freeze and gates fresh training and P6 runtime qualification; it
does not gate isolated P1 authoring or pre-G0 source/interface disposition.

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

### G0: runtime, integration, and promotion release

G0 is not a prerequisite for isolated source authoring or static source/interface
review after accepted P0a, the accepted overall plan, and this exact P1 plan
revision's Pass 1. Those tasks use a separate HDD worktree/branch and must not access or
modify the campaign checkout, dashboard, workers, checkpoints, outputs, leases,
or devices. While supervisor PID 2750253 is live, the allowed activity is
limited to source inspection, source-only edits within this plan's allowlist,
static review of pinned diffs/interface dispositions, and documentation/hash
checks that do not execute project code.

Before running any test/build/import of project code, GPU/device work, simulator,
dashboard operation, branch integration, merge, promotion, or final acceptance,
the user must manually stop the campaign and both WX9100 leases must be
independently confirmed released by their lease owners. Never stop or signal the
campaign as part of this plan. Do not infer G0 from elapsed time, missing
telemetry, an absent process observation, or device utilization. Record the
manual stop evidence and each released device UUID/lease identity in the
post-G0 acceptance record. Until then, no integration checkout or publication
of the candidate API/ABI is permitted.

### P0a: approved preregistration before source work

P0a is an upstream dependency for source discovery and implementation. This
plan does not create or approve it. Before any source inspection or edit under
the implementation plan, P0a must be immutable, independently reviewed, and
carry the required human decision/approval provenance. Its digest and exact
URI/path must be pinned in the implementation ledger. It freezes the
source-independent decisions and methodology:

- The accepted `simulated_match_score` definition and scoring-rule boundaries:
  accepted active-HUB FUEL awards only, separate from official match points.
- The preregistered meaning of one score event and `score_units`; any additional
  event type or unit requires an amended V4 contract.
- The exact scenario and seed partition bytes, calibration sample design,
  calibration method, metrics, margins, and pass/fail predicates, fixed before
  inspecting candidate outcomes. P0a records seed IDs/hashes but makes no
  source or calibration-result claim.
- The source/rules identity fields P0b must populate: exact source commit and
  dirty-tree state, source-path/symbol map, source identity digest, P0a and
  rules artifact hashes, simulator/config/runtime identity, and approval
  provenance. P0a may declare required fields without guessing their values.
- Identity-domain constraints: physical simulator piece slot/generation is
  disjoint from observer-track handles. P0a does not guess how the base
  allocates piece generations or handles reset/reuse.

No accepted P0a artifact or frozen seed design was supplied by this task. P0a
must be accepted before P1.1 starts. If P1.1 discovers that the registered
estimand or event boundary cannot be traced to authoritative source, stop
score-dependent implementation and return to the P0a owner for reviewed
amendment; do not reinterpret the score after seeing outcomes.

### P0b: final source-bound calibration after source freeze

P0b is produced only after all P1–P6 source implementation and integration
are complete, frozen, and independently reviewed, and final source/config/
runtime identity is known. P1.1’s source map is a required input. P0b fills
P0a’s source/rules identity fields, runs the preregistered calibration method
on that frozen source, and emits the immutable Phase 0 artifact and target-
scoped protocol with exact hashes and required human decision/approval
provenance. Under the
accepted overall plan, P6 owns the qualification schema, runner, and associated
source/tests; P0b owns only the post-freeze calibration operation and generated
immutable artifacts. P0b does not wait for P6 runtime qualification. P0b may
not change P0a’s estimand, scenarios, seed partitions, calibration method,
margins, or pass predicates after observing results. A result that misses a
preregistered gate blocks
fresh training or P6 runtime qualification; source/rules changes require a
new source freeze and P0b artifact. Fresh training and P6 runtime qualification remain
prohibited until P0b is accepted and bound to the exact implementation source.

The `uint16` episode-sequence bound, event ordering, tick meaning,
`match_elapsed_seconds`, alliance/robot-slot mapping, physical piece generation
creation/increment/reset behavior, and authoritative producer/consumer map
must be established from P1.1 source evidence and attached to P0b. The later
owner response's recorder/module citation is not assumed to exist in this
base. If the source map contradicts P0a or the accepted V4 registry/capacity,
stop and request a P0a/V4 reconciliation before proceeding.

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
evidence for this checkout. After accepted P0a, P1.1 must search the pinned
tree, imports, package metadata, generated/native dependencies, and relevant
history, and return an exact source map. That map feeds P0b's final
source/rules artifact and must close before P0b is accepted. If it does not
locate the authoritative episode event source/consumer or resolve the cited
source mismatch, stop for a P0a/plan amendment before score-dependent
implementation. Do not expand the allowlist to an unreviewed path.

## Finite package sequence

The work is sequential. Every package produces a pushed commit and exact-head
review before its dependent package starts. A REVISE returns only that package
to its owner; a changed schema, profile, event code, source contract, or byte
total stops the sequence pending reconciliation.

### P1.1 — Read-only source map and interface decision record

**Dependency:** accepted P0a preregistration, accepted overall plan, and this
P1 plan's exact-head Pass 1 ACCEPT. G0 is not required for source inspection
or this static record. No tests, builds, imports, runtime, device, simulator,
dashboard, integration, merge, or promotion. This read-only source map is
required input to P0b; it is not gated on the final calibrated P0b artifact.

Trace each producer, mutation, reset, and consumer from the base tree. Record
the exact path/line, device/dtype/shape, source event, physical piece identity,
source-site branch, reset boundary, and current transfer/persistence path.
Resolve all still-open row-level meanings that do not change V4 bytes,
including deterministic valid-row ordering and how the 2,304 reserved slots
are filled across a 32-row batch. The record must distinguish accepted owner
decisions from observed source behavior. If the chosen interpretation changes
any field, type, identity domain, code, capacity, offset, or byte, stop for
reconciliation.

**Output:** an implementation-ledger entry containing the P0a
identifier/hash, source map, V4 profile hash, exact current base, per-file
owner, producer/consumer interface, and an explicit resolve/hold decision for
each open item above. Deliver the source map and populated source/rules
identity fields to the P0b owner. The map is an input to the final artifact,
not itself calibration or P0b approval.

### P1.2 — V4 typed schema, encoding, digests, and strict reader

**Dependencies:** accepted P0a preregistration, P1.1 decision record, and
immutable profile pins. P0b is not a P1 implementation or test prerequisite;
it follows the P1–P6 source freeze and gates fresh training and P6 runtime
qualification.

Implement the typed field declarations and exact row/batch layout, versioned
96-byte header, checked offset arithmetic, little-endian field packing,
device-side validity checks and digest computation, and an exact-length
version-first reader. Keep the 59-byte record packed without implicit native
struct alignment. The reader rejects unsupported versions before parsing
payload fields; mismatched profile, lengths, codes, digests, reserved bits,
invalid valid-fields, or overflow reject the whole batch without returning a
validated typed view. Caller-side no-mutation behavior belongs to P5 tests.
Unused event records are fully zeroed apart from `valid=0`. Persist the exact opaque bundle; no host-side
repair, hash, decode, aggregation, or selection of live payload values.

**Output/interface:** a standalone GPU-resident V4 batch and manifest API,
with a strict reader that returns validated typed views only after version,
length, digest, profile, registry, and mask checks. The artifact writer
returns exact immutable bundle bytes and manifest identity; it may stage a file
write but may not publish a partial bundle. This is the API boundary P5 will
consume after this P1 API head is independently accepted. P1 does not modify
trainer callers or their integration tests.

### P1.3 — Physical score identity and device-native event staging

**Dependencies:** accepted P0a preregistration, P1.1 source map, and P1.2
schema interfaces. Do not implement a score-dependent change if observed
source behavior conflicts with the preregistered rule; return to P0a/V4
reconciliation first. The final P0b artifact is produced after the resulting
source is frozen.

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

**Output/interface:** the simulator exposes only typed staged V4 score rows
plus explicit failure status through the standalone API. The API exposes the
same committed event rows used by aggregate score projection; it must not
reconstruct physical piece identity from aggregate `fuel_scored_event`
counters. Caller-side trainer integration is owned by P5.

### P1.4 — Consumer, bundle assembly, and checkpoint/manifest binding

**Dependencies:** accepted P1.2 and P1.3 interfaces.

Assemble and validate the standalone V4 batch from the transition and event
tensors at the P1-owned simulator/API seam; expose the source/rules-bound score
projection through the API identified in P1.1. Prove that event count and
event-derived `simulated_match_score` agree with the preregistered P0a rule,
while route/risk shaping and official-score representations remain separate.
P1 does not insert this batch into a trainer rollout buffer. P5 alone connects
the accepted API to shared trainer callers and owns those integration tests.
Bind the profile, source, configuration, checkpoint, track, schema/envelope,
protocol, and payload digest fields exactly as adopted. Use a versioned V4
manifest/reader contract; the base's schema-version-1 checkpoint or any JSON
metric ledger is not a V4 substitute. Preserve atomic publication and
all-or-nothing state handling. Do not change PPO reward, GAE, terminal,
truncation, optimizer, curriculum, or model behavior in this package.

**Output/interface:** a validated standalone transition batch and immutable
matching manifest, linked by exact source/config/checkpoint/profile/protocol
identity and byte digests. This exact API head is a P5 dependency; P5 consumes
it from the shared trainer/evaluation caller without P1 edits to that caller.

### P1.5 — Exact-head review, seeded simulator and dashboard evidence

**Dependencies:** P1.1–P1.4 have pre-G0 source/interface disposition; G0 has
been independently evidenced; accepted P0a remains pinned; the reviewed source
commits are integrated into a fresh current checkout only after G0. This is
post-G0 implementation regression evidence, not final P0b calibration or
training/qualification.

Run only the package-authorized schema/event tests and representative
end-to-end simulator evaluations. Use the P0a-frozen scenario/seed IDs; include
score-producing scenarios exercising each supported score-capture branch
without changing the event rule. Inspect the fresh simulator record/replay:
valid events must tie to the physical piece incarnation and actual accepted
score mutation, unused rows must remain zero, aggregate output must match the
P0a estimand, and failure injection must leave no partial score/event/artifact
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
campaign, checkpoint, or data path is authorized by this plan. Shared trainer
caller code (`frc_defense/meco_track_training.py`) and all trainer
integration tests are expressly excluded and belong solely to P5.

| Path | P1 authority |
|---|---|
| `frc_defense/meco_transition_v4.py` | New V4 field registry, fixed-layout encoder/reader, validation, and digest interfaces. |
| `frc_defense/meco_score_events_v4.py` | New device-resident score-event staging/validation and batch assembly helpers. |
| `frc_defense/gamepiece_actions.py` | Capture and stage only at proven score-mutation sites; preserve existing scoring behavior. |
| `frc_defense/_tensor_3v3_gamepieces.py` | Physical score-generation lifecycle only after source discovery resolves initialization/reset ownership. |
| `frc_defense/tensor_3v3.py` | V4-scoped event/generation buffers, reset integration, and typed transition output seam only. |
| `tests/test_meco_transition_v4.py` | New schema, exact-byte, digest, length, and reader fixtures. |
| `tests/test_meco_score_events_v4.py` | New event ordering, identity/generation, reset, capacity, and atomic-failure fixtures. |
| `tests/test_meco_transition_v4_simulator.py` | New seeded end-to-end standalone producer/artifact and score-mutation/replay fixture; it must not invoke trainer callers. |
| `tests/test_fuel_physics_integration.py` | Existing integration assertions only if required for source-site behavior coverage. |
| `tests/test_tensor_3v3_physical_constraints.py` | Existing scoring regression assertions only if required for source-site behavior coverage. |

`frc_defense/meco_simulated_score_ledger.py` is intentionally not on the
allowlist: the path is not present in the pinned base checkout. P1.1 must
establish whether an authoritative source is present elsewhere or whether P0a
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
   capacity exhaustion. Any bound not proven by P1.1 source evidence and
   attached to P0b remains a blocked fixture,
   not a fabricated expected value.
4. Fault injection before and during staging, before score-state commit, and
   before artifact rename/publication. Assert the specified no-op state for
   score counters, piece ownership/respawn, event table, and artifact
   publication. Optimizer, RNG, checkpoint, and other trainer-caller invariants
   are tested by P5 under its exclusive trainer ownership. If a physical tick
   already mutated and cannot be rolled back, P1 must demonstrate staging of
   the package-owned score mutation before that boundary or halt for redesign.
5. A byte-identical GPU-produced bundle and digest against fixed independent
   fixtures. Any host reference encoder is test-only and may not enter the
   live producer, digest, reader, or decision path.

**Final post-G0 acceptance evidence must include:** exact P0a preregistration URI/hash and
approval provenance; exact source map; accepted P1.1 decision record; exact base and
profile hashes; per-package pushed commits and file allowlist; all fixture
results; GPU UUID/runtime evidence for both WX9100s; no-CPU transfer/sync trace
for the event-to-artifact path; and exact artifact/manifest hashes. Pass 3 must
re-open the exact pushed head/blob, review all allowlist changes and fixture
results, and return a persisted ACCEPT/REVISE. No mutable branch-tip review is
accepted.

**Pre-G0 Pass 3 is source/interface disposition only.** It may inspect the
exact pushed isolated candidate, its source diff, schema/interface consistency,
allowlist, and static provenance. It must not run code, fixtures, tests, builds,
imports, or runtime checks. Record its result as `PRE_G0_SOURCE_INTERFACE_ACCEPT`
or `REVISE`; that result is not P1 completion, implementation validation, merge
approval, promotion, or evidence of simulator/GPU/dashboard correctness. Any
source change after that disposition requires another exact-head pre-G0 review.
After G0, perform P1.5's required tests, both-WX9100 no-CPU evidence, seeded
simulator/replay evaluation, and current dashboard/backend/frontend checks.
An independent reviewer then reopens the exact integrated head and those
runtime records for `POST_G0_FINAL_ACCEPT` or `REVISE`. Only the latter can
close P1.

For every test that checks scoring or other simulated behavior, evidence must
also contain the representative seeded simulator evaluation and inspected
record/replay. Report each test name, scenario, exact seed, replay identifier,
result, device UUID/runtime, and whether dashboard backend/frontend freshness
and endpoints were verified. If any required simulator or dashboard gate
cannot run, mark validation incomplete; a unit/regression pass is not an
end-to-end pass.

### P0b closure after P1–P6 source freeze

After all P1–P6 source implementation and integration are complete, frozen,
and independently reviewed, the P0 owner/operator runs the P0a preregistered
calibration against that frozen source and emits the final source/rules-bound
P0b artifact. P1.1’s producer/consumer map and exact final source/config/runtime
identity are required inputs, along with the preregistered scenario/seed and
calibration outputs and required independent review and human approval
provenance. P6 owns the qualification schema, runner, and
source/tests used by later qualification; P0b does not own or edit those
source files and does not depend on P6 runtime qualification. Pass criteria
and seed partitions cannot be revised after seeing calibration results. A
source change after P0b invalidates its source binding and blocks fresh
training and P6 runtime qualification until the calibration artifact is
renewed and accepted. P0b is not a prerequisite for P1 implementation, tests,
or simulator/dashboard regression evidence.

## Risks and explicit open decisions

- **P0 sequencing:** No accepted P0a preregistration was supplied. P1 source
  discovery and isolated source authoring require accepted P0a and the exact
  plan prerequisites; tests, builds, runtime validation, integration, merge,
  promotion, and final acceptance require G0. P1.1 feeds source identity to P0b. P0b closes after all P1–P6
  sources are frozen/reviewed and gates fresh training and P6 runtime
  qualification, not P1 source work or validation.
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
- **Package boundary:** P1 publishes the standalone typed batch/manifest API
  and producer/artifact fixtures. P5 is the sole owner of
  `frc_defense/meco_track_training.py` and trainer integration tests. P5 must
  depend on this exact P1 API head after independent acceptance; any API change
  after that pin requires P5 to re-review/rebase before integration. P1 must not
  edit shared trainer callers or tests. `gamepiece_actions.py` and
  `tensor_3v3.py` may have other owners; reserve only the scoped P1 hunks or
  defer conflicting edits through the package coordinator.

## Completion and non-goals

P1 implementation and validation are complete only after G0, when the exact reconciled
profile is implemented for schema, score-event producer and consumer, and
bundle/manifest; exact-byte, digest, failure, and source-identity fixtures
pass; required seeded simulator records/replays show the changed scoring
behavior; both assigned GPUs are verified; and the current dashboard
backend/frontend pass freshness and endpoint checks. These P1 gates depend on
accepted P0a and G0, not P0b. A pre-G0 source/interface ACCEPT is only a review
checkpoint and cannot be reported as P1 completion. Integration and merge are
allowed only after manual stop and independent release of both WX9100 leases.
P1 completion supplies frozen source and the
source map to P0b. Under the accepted overall plan, P0b closes after all P1–P6
sources are frozen and reviewed, then gates fresh training and P6 runtime
qualification. Neither training nor P6 runtime qualification is authorized by
this P1 plan.
The P1 package does not qualify A1/A2/A3/C1/C2, authorize continuous training,
publish V4 as a repository-wide normative ABI, or replace subsequent PPO,
curriculum, model, GPU-residency, and Stage 1 plans.

No source change, test, build, simulator, GPU, dashboard, checkpoint,
campaign, or training action was performed while preparing this plan.
