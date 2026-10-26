# P1.2 standalone transition-row-block codec plan

**Date:** 2026-10-26
**Status:** Pass 1 plan candidate; full-bundle work is blocked
**Planner:** `/root/p1_p2_planner`
**Package:** P1.2 transition-row-block serialization only

## Purpose and boundary

The accepted P1.2 schema is sufficient to plan a standalone codec for the 32
transition rows. It is not sufficient for a full V4 bundle codec: the accepted
proposal does not give the score-event table's payload offset/order. The
separate owner-amendment proposal
`docs/plans/meco-v4-payload-section-order-owner-amendment-20261026.md` lists
exact layout alternatives but has no adoption decision. Until one layout is
adopted and re-pinned, this slice must not assemble or parse the complete
payload.

This plan covers device-side pack/unpack and structural shape/type validation
of the transition-row byte block only. It does not implement a bundle header,
identity prefix, score-event placement, payload digest, full-bundle reader,
opaque artifact persistence, event production/consumption, or runtime
integration. A row block produced here is not a complete V4 bundle and must
not be represented as V4 artifact compliance.

Static source and test-fixture authoring may proceed before G0 only after this
exact plan receives independent Pass 1 ACCEPT. Imports, test execution,
builds, device operations, simulation, dashboard, training, integration,
merge, and promotion remain behind the accepted G0 and reviewed bridge gates.

## Exact accepted inputs

- P1 successor plan: commit
  `40d151dec437fe405bd84d2918b2b86ae6f4a129`, plan SHA-256
  `1cfe7ccd56a4355501a7e46a3d3e3cd9bf0ae1c443bb0da571c27e05f6043300`;
  Pass 1 ACCEPT record commit
  `2810a315970c56b0161bbfbae025a7a0301ece8e`, record SHA-256
  `dedcca460b1cfc7d89c2f13ccd383783f674a5085f99052c4e30caf21f4ba339`.
- Approved implementation base: `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.
- Accepted transition schema: commit
  `38fdb1d0423f0e723c4e1fb533e36eb7b88c4715`, file
  `frc_defense/meco_transition_v4.py`, SHA-256
  `689b84acfd2d327c9ca4c26b081cf997acf511a0f14ed6f605a6caa91562f52f`;
  SOURCE/API ACCEPT review commit
  `2cc1275a9cbf3949419cb4925cb1f2b1fb208f85`, record SHA-256
  `3352d0c7a4963a928c8e5f4ab0bc9237647e57459963abf9a933f312f0c58b25`.
- Accepted score-event registry is a read-only dependency: commit
  `78b640630fde1aa9c0934463525e322ba57e027e`, file SHA-256
  `5e4c803a676e7a445b686d0f73e5d18e7341533119571eae8e9545d7a259d278`;
  SOURCE/API ACCEPT review commit
  `abc4a3d712bf829d0af5ec7a3ece9e173f7039c3`.
- Accepted profile byte totals: 32 rows × 9,822,906 = 314,332,992 bytes.
  The accepted static identity prefix is 123 bytes and precedes row 0, so the
  row block follows that prefix at payload-relative offset 123 in the proposed
  layouts. This slice returns only the row block, never the payload.

Start implementation at the exact accepted schema head
`38fdb1d0423f0e723c4e1fb533e36eb7b88c4715` in a fresh ZFS worktree. This
preserves the accepted base and both reviewed declarations without rebasing
onto later `main`.

## Transition-row byte order

The accepted proposal says each field column repeats for 32 rows and fields
inside a group concatenate in listed order. The declaration groups and field
lists are authoritative. This slice makes traversal concrete as follows:

1. Visit groups in `TRANSITION_FIELD_GROUPS` order.
2. Within each group, visit fields in `TRANSITION_FIELDS` order.
3. For a field, visit `rollout_row_index` from 0 through 31.
4. For each row's shaped field, emit elements in C order: the last array axis
   varies fastest, then preceding axes.
5. Emit each scalar using the declared little-endian type width, with no
   alignment or padding. Concatenating all groups yields exactly 314,332,992
   bytes.

The event table is absent from the returned byte block. Its placement and
internal table order are not inferred here.

## Float and mask handling

This block codec preserves input IEEE-754 float bit patterns and boolean
values exactly after the device-side layout conversion. It does **not** test
whether valid floats are finite, whether mask-false float values are zero,
whether any mask is semantically correct, or whether any source value is
otherwise V4-valid. Those accepted profile rules remain upstream producer/API
responsibilities. The codec must not zero, normalize, clamp, canonicalize, or
repair float/bool fields. It performs no host-side numeric inspection. Its
acceptance claim is limited to byte layout and field shape/type fidelity, not
complete V4 data compliance.

## Finite source allowlist and API

Only these files may change:

| Path | Allowed change |
|---|---|
| `frc_defense/meco_transition_v4.py` | Add device-only `pack_transition_rows_device`, `unpack_transition_rows_device`, and structural input/output checks for the exact transition block. Reject non-device input and wrong shapes/dtypes/lengths. No header, identity, event table, payload digest, bundle APIs, or persistence helpers. |
| `tests/test_meco_transition_v4.py` | Author exact block-length, group/field/row/axis order, field-type, and raw float/bool round-trip fixtures. Test authoring only pre-G0; tests must target the assigned devices post-G0. |

Do not edit `frc_defense/meco_score_events_v4.py`; it is a read-only dependency.
No new native HIP/C++ source, build configuration, package dependency,
trainer hook, environment/gamepiece source, campaign, checkpoint, output,
dashboard, or artifact file is allowed. Any unlisted-path need stops for a
reviewed allowlist amendment.

## G0 and validation gates

### Before G0

After exact-head Pass 1 ACCEPT, only source/test-file authoring and static
review are permitted. Do not import project code, run a test, build/compile,
touch a GPU/device, run simulator/dashboard/training, mutate checkpoints,
integrate/merge/promote, or inspect/operate the active campaign or device
leases. Pre-G0 reviewer checks only the exact diff, field traversal order,
device-only paths, no host numeric/hash use, no semantic float/mask checks,
and the two-path allowlist.

### After G0 and reviewed contract bridge

Execution may start only after all accepted G0 evidence exists: the user
manually stopped campaign PID `2750253`; both assigned WX9100 leases are
independently confirmed released by their owners; and the user or repository
maintainer explicitly authorized runtime/device/integration work. The
accepted P1 plan additionally requires the reviewed contract-anchor bridge
before validation/integration.

Run the exact codec tests on each assigned WX9100. Record device UUID/runtime
identity, device placement, memory ownership, transfers/synchronization, and
proof that no numeric bytes moved to CPU for inspection. CPU/P600 fallback or
silent migration fails closed. This slice's tests inspect byte layout only
and do not exercise simulated robot/FUEL behavior, so they do not claim a
simulator evaluation or dashboard replay. Any later test that checks
simulated behavior must include the representative end-to-end simulator and
current-dashboard checks required by project `AGENTS.md`.

Full-bundle serialization, payload SHA-256, envelope parsing, and artifact
write/read remain blocked until an owner/maintainer adopts a payload section
order, the proposal/profile bytes are re-pinned, and a successor exact-head
SOURCE/API review accepts the result. P1.3 producer and P1.4 consumer/artifact
work remain independently blocked on event semantics and their own gates.

## Roles

- Planner: `/root/p1_p2_planner`.
- Implementer: `/root`.
- Pre-G0 SOURCE/API reviewer: `/root/p3_p4_planner`.
- Post-G0 validation/final reviewer: `/root/p0a_implementer`.

Each reviewer is distinct from planner and implementer. This candidate does
not authorize source edits until its exact-head Pass 1 ACCEPT. No source,
tests, imports, builds, simulation, dashboard, GPU/device, campaign, lease,
checkpoint, or artifact operations were performed to prepare this plan.
