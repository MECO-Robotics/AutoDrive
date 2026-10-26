# P1.2 fixed-layout codec, device checks, and opaque persistence plan

**Date:** 2026-10-26  
**Status:** Pass 1 implementation-plan candidate; planning only  
**Package:** P1.2 V4 fixed-layout codec and bundle persistence  
**Planner:** `/root/p1_p2_planner`

## Feasibility and boundary

The accepted P1 finite allowlist can support this bounded slice without making
event semantics normative. The existing transition-schema module can encode
and decode the accepted byte layouts and expose device-side structural and
digest operations. The event table is an opaque, fixed-size byte region at
this layer: the codec preserves its 2,304 × 59 bytes without deriving event
identities or checking producer ordering. A separate host helper may write or
read the completed bundle as opaque bytes without inspecting, hashing,
repairing, or numerically transforming it.

This plan authorizes source authoring after its own exact-head Pass 1 ACCEPT.
Static authoring and review may occur before G0. It does not authorize imports,
tests, builds, execution, device access, simulator evaluation, dashboard work,
training, integration, merge, or promotion. Runtime correctness and GPU
placement remain unverified until all accepted G0 conditions are met.

## Authority and exact inputs

- Accepted P1 successor plan: commit
  `40d151dec437fe405bd84d2918b2b86ae6f4a129`, file SHA-256
  `1cfe7ccd56a4355501a7e46a3d3e3cd9bf0ae1c443bb0da571c27e05f6043300`;
  Pass 1 ACCEPT record commit
  `2810a315970c56b0161bbfbae025a7a0301ece8e`, record SHA-256
  `dedcca460b1cfc7d89c2f13ccd383783f674a5085f99052c4e30caf21f4ba339`.
- Pinned implementation source base: `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.
- Accepted corrected P1.1 score source map: commit
  `c902659d1b9079b3afcc2e4796780d97020b6726`, file SHA-256
  `c8c1d0594977ec456d8513293fe7ba0ca0020eba52a822e75cad34104aa36b4e`.
- Accepted P1.2 score-event field registry: commit
  `78b640630fde1aa9c0934463525e322ba57e027e`, file
  `frc_defense/meco_score_events_v4.py`, SHA-256
  `5e4c803a676e7a445b686d0f73e5d18e7341533119571eae8e9545d7a259d278`;
  exact-head SOURCE/API ACCEPT review commit
  `abc4a3d712bf829d0af5ec7a3ece9e173f7039c3`, record SHA-256
  `47d1426ea8b02821a9521839d8c753a2e9e9a8d7505a42a1480fa615a1e7dff5`.
- Accepted P1.2 transition/bundle schema registry: commit
  `38fdb1d0423f0e723c4e1fb533e36eb7b88c4715`, file
  `frc_defense/meco_transition_v4.py`, SHA-256
  `689b84acfd2d327c9ca4c26b081cf997acf511a0f14ed6f605a6caa91562f52f`;
  exact-head SOURCE/API ACCEPT review commit
  `2cc1275a9cbf3949419cb4925cb1f2b1fb208f85`, record SHA-256
  `3352d0c7a4963a928c8e5f4ab0bc9237647e57459963abf9a933f312f0c58b25`.
- Reconciled V4 artifact: commit
  `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, SHA-256
  `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`;
  profile SHA-256
  `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`;
  raw proposal/profile digest
  `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.

Implement from a fresh ZFS worktree at accepted source head
`38fdb1d0423f0e723c4e1fb533e36eb7b88c4715`. Its ancestry includes the exact
approved base and both independently accepted declarations above. Do not
rebase onto current `main` or copy later source onto the pinned base.

## Fixed implementation scope

The codec must consume only the accepted declarations and reconciliation
artifact. Preserve packed little-endian layout, field order, offsets, shapes,
track codes, fixed capacities, and exact byte counts:

- 96-byte v2 header and 123-byte static identity prefix.
- 32 transition rows, 9,822,906 bytes each.
- 2,304 score-event rows, 59 bytes each, passed through as opaque bytes.
- 314,469,051-byte payload and 314,469,147-byte complete bundle.
- The adopted profile digest, payload digest span, reserved-zero fields,
  fail-closed version-first reader rule, and exact-length predicate.

The implementation may add these schema-layer operations in
`frc_defense/meco_transition_v4.py`:

1. Device-side pack/serialize of fields in declared order to one contiguous
   uint8 bundle; no host numeric conversion or implicit padding.
2. Device-side read/deserialize of the same byte layout, rejecting an invalid
   version/header or non-exact bundle length before accepting payload fields.
3. Device-side structural validation and SHA-256 digest operations over the
   exact accepted byte spans, using only existing device-native tensor
   primitives. Field shape, dtype, row count, byte length, fixed codes,
   reserved bytes, and adopted digest spans may be checked. No numeric tensor
   or digest input may materialize on or be checked by CPU code.
4. A host-only opaque persistence boundary that transfers completed uint8
   bytes to/from storage unchanged. It must not decode fields, compute
   digests, repair bytes, or transform numeric values on the host.

Treat the score-event table as an uninterpreted fixed byte table in these
operations. Preserve all rows, including unused rows. Do not add or enforce
event sequence allocation/order, generation lifecycle, tick sampling, score
mutation, event uniqueness, or ledger projection rules. Those belong to the
blocked producer/consumer work and accepted owner decisions. If exact digest
domain details or a required wire behavior cannot be established from the
pinned accepted contracts, stop and request a reviewed contract amendment;
do not infer a digest domain or alter schema bytes.

No new native HIP source, build configuration, dependency, trainer hook,
dashboard, campaign path, checkpoint path, or runtime output path is in scope.
If static API design cannot establish a complete device-resident operation
path using existing tensor primitives, stop and request a reviewed allowlist
amendment; do not add a native kernel or CPU fallback under this plan.

## Finite allowlist

Only these paths may change in this slice:

| Path | Allowed change |
|---|---|
| `frc_defense/meco_transition_v4.py` | Device-side fixed-layout serializer/reader, structural and digest operations, plus opaque byte persistence helpers. |
| `tests/test_meco_transition_v4.py` | Author exact layout, boundary, malformed-header/length, digest-span, and opaque-byte round-trip fixtures. Test authoring only before G0; test execution is post-G0. |

`frc_defense/meco_score_events_v4.py` is a pinned read-only schema dependency
in this slice. Do not edit it. The accepted P1 plan reserves the following
work for later gates: score-event production/mutation in P1.3; semantic event
validation and event-derived score projection; full source/manifest binding
that needs P5 data; and all trainer changes. No edits to
`frc_defense/meco_track_training.py`, gamepiece producers, environment reset
or step logic, metric/evaluation consumers, HIP/C++ sources, build files,
dashboard files, campaign data, checkpoints, or outputs are authorized. An
unlisted-path dependency is a stop condition requiring reviewed allowlist
amendment.

## Source-authoring and acceptance sequence

1. This exact plan receives independent Pass 1 ACCEPT. Planner, implementer,
   and pre-G0 reviewer remain distinct.
2. Implementer starts at exact head `38fdb1d0423f0e723c4e1fb533e36eb7b88c4715`
   in a fresh ZFS worktree. The accepted P1.2 registry hashes and source base
   above are immutable inputs. Source authoring and test-fixture authoring may
   proceed pre-G0 within the two-path allowlist.
3. Before G0, perform static diff and source review only. Reviewer checks
   exact field ordering, byte offsets/lengths, device-native tensor operations
   for dynamic bytes and hashing, no host numeric/hash operations, no
   event-semantic assumptions, no CPU/P600 fallback, and no out-of-scope edits.
   If the device-resident path cannot be established statically, fail closed
   for a reviewed allowlist amendment. Do not import project modules or execute
   tests/builds.
4. After G0, run the exact codec tests and verify both WX9100 UUID/runtime
   identities, device residency for pack/validation/digest operations,
   transfer boundaries, no CPU/P600 fallback or migration, and exact opaque
   persisted bytes. Fail closed if any dynamic bytes or digest input are
   evaluated on CPU or if an operation silently migrates devices; request a
   reviewed allowlist amendment if the existing primitives cannot maintain
   device residency. Static source acceptance alone is not runtime evidence.
5. Because these tests exercise V4 serialization and score-event bytes, run a
   representative end-to-end simulator evaluation under the project
   `AGENTS.md` rule. Record scenario, seed, replay/result identity, and outcome.
   Before replay review, restart and verify the current dashboard backend and
   frontend, generate a fresh replay, and confirm its provenance. If any G0
   condition or device path is unavailable, stop and report that validation
   did not run.
6. Keep P1.3/P1.4 blocked. This slice does not integrate the codec into score
   producers, trainer, or campaign and does not claim P1 or package completion.

### G0 execution gate

No test, import, build, GPU/device operation, simulator, dashboard, training,
checkpoint mutation, integration, merge, or promotion may begin until all are
recorded: (1) the user manually stopped campaign PID `2750253`; (2) both
assigned WX9100 leases are independently confirmed released by their owners;
and (3) the user or repository maintainer explicitly authorized runtime,
device, and integration work. The accepted P1 plan additionally requires the
reviewed contract-anchor bridge before validation/integration. Preserve the
GPU-only numerical rule: both assigned WX9100s are required; CPU numerical or
digest work, P600 substitution, and silent migration are prohibited. CPU is
limited to supervision, configuration, and unchanged opaque I/O. Retain
per-process device identity, kernel/device placement, memory ownership, and
transfer/synchronization evidence for each device.

## Roles

- Planner: `/root/p1_p2_planner`.
- Implementer: `/root`.
- Pre-G0 SOURCE/API reviewer: `/root/p3_p4_planner`.
- Post-G0 validation/final reviewer: `/root/p0a_implementer` (distinct from
  planner, implementer, and pre-G0 reviewer).

The plan does not authorize source edits until its exact-head Pass 1 ACCEPT.
It does not claim runtime, simulation, dashboard, digest, or device-residency
success. No tests, imports, builds, simulator, dashboard, GPU/device,
campaign, lease, checkpoint, or output operations were performed to prepare
this plan.
