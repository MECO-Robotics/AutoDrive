# Proposed V4 payload section-order owner amendment

**Date:** 2026-10-26  
**Status:** Decision proposal only; no layout choice is adopted  
**Planner:** `/root/p1_p2_planner`  
**Purpose:** Resolve the missing byte placement of the fixed score-event table
before full-bundle codec, payload-digest, reader, or artifact-persistence work.

## Accepted evidence and the gap

The accepted V4 proposal fixes the 96-byte header, 123-byte static identity
prefix, 32 transition rows of 9,822,906 bytes each, and 2,304 score-event
records of 59 bytes each. It fixes payload length at 314,469,051 bytes and
complete bundle length at 314,469,147 bytes. It says the static identity
fields precede transition row 0 and describes the payload as a packed
field-order byte stream. The reconciled artifact independently pins the same
sizes and encoding.

The accepted proposal and owner reconciliation do **not** state the score
event table's payload offset or whether that table precedes or follows the
transition row stream. They also do not say whether the identity prefix must
be adjacent to row 0. Byte counts alone cannot decide the wire layout. Because
the payload digest hashes the exact payload bytes, full-bundle packing,
reading, digest verification, and persistence cannot safely begin until this
gap is adopted and pinned.

Exact authority pins inspected:

- V4 proposal commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141`, SHA-256
  `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- Accepted reconciliation commit `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`,
  artifact SHA-256
  `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`,
  profile SHA-256
  `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`.
- Accepted transition schema declaration `38fdb1d0423f0e723c4e1fb533e36eb7b88c4715`,
  file SHA-256
  `689b84acfd2d327c9ca4c26b081cf997acf511a0f14ed6f605a6caa91562f52f`,
  exact-head SOURCE/API ACCEPT review `2cc1275a9cbf3949419cb4925cb1f2b1fb208f85`,
  record SHA-256
  `3352d0c7a4963a928c8e5f4ab0bc9237647e57459963abf9a933f312f0c58b25`.
- Accepted score-event field registry `78b640630fde1aa9c0934463525e322ba57e027e`,
  file SHA-256
  `5e4c803a676e7a445b686d0f73e5d18e7341533119571eae8e9545d7a259d278`,
  exact-head review `abc4a3d712bf829d0af5ec7a3ece9e173f7039c3`, record
  SHA-256 `47d1426ea8b02821a9521839d8c753a2e9e9a8d7505a42a1480fa615a1e7dff5`.

## Proposed choice A: identity, transitions, events

This is the recommended amendment because it keeps the 123-byte static
identity prefix immediately before transition row 0 and places the fixed event
table after the ordered transition rows.

Offsets are payload-relative; the 96-byte envelope header is outside the
payload:

| Payload section | Start | End (exclusive) | Bytes |
|---|---:|---:|---:|
| Static identity prefix | 0 | 123 | 123 |
| Transition field stream | 123 | 314,333,115 | 314,332,992 |
| Score-event table | 314,333,115 | 314,469,051 | 135,936 |
| **Payload end** |  | **314,469,051** | **314,469,051** |

The complete bundle then has header offsets `[0,96)`, payload offsets
`[96,314,469,147)`, and total length 314,469,147 bytes. The transition row
stream has 32 rows × 9,822,906 bytes. The accepted profile says each column
repeats for 32 rows and fields within each group concatenate in listed order.
This proposal makes the traversal explicit: for each field in the accepted
group/field order, emit rows in increasing rollout-row index, then emit each
field's array axes in canonical C order (last axis fastest). The event table
is 2,304 consecutive 59-byte records in increasing table-slot index; within
each record, emit the accepted 13 fields in registry order and offset. This
ordering defines byte
placement only; it does not choose event production or same-tick sequence
semantics.

## Alternative choice B: identity, events, transitions

If the owner wants the event table adjacent to the static identity instead,
the exact layout would be:

| Payload section | Start | End (exclusive) | Bytes |
|---|---:|---:|---:|
| Static identity prefix | 0 | 123 | 123 |
| Score-event table | 123 | 136,059 | 135,936 |
| Transition field stream | 136,059 | 314,469,051 | 314,332,992 |
| **Payload end** |  | **314,469,051** | **314,469,051** |

All transition rows and event records retain the same internal orders stated
for choice A. The identity prefix still precedes row 0 but is not adjacent to
it. The 96-byte header and total bundle size are unchanged.

The owner may reject both proposals and specify another contiguous layout,
but the response must list every section's start, exclusive end, length, and
internal order. No padding, gaps, overlap, hidden sidecar, or trailing bytes
are permitted under this amendment.

## Exact owner/maintainer decision requested

An authorized Artifact/manifest and shared-ABI owner, with maintainer
ratification, must record one of:

1. **ADOPT A** exactly as proposed;
2. **ADOPT B** exactly as proposed; or
3. **REVISE** with a complete alternative section table and serialization
   order satisfying the same fixed field inventory and byte totals.

The response must also confirm:

- Static identity prefix placement relative to row 0 and the event table.
- Transition field-major flattening: field/group order, increasing row index,
  and last-axis-fastest C-order inner axes.
- Event-table record-major order by table slot, then registry field order.
- Payload digest preimage is exactly the contiguous payload section from byte
  0 through byte 314,469,050, excluding the envelope header.
- Profile digest remains SHA-256 of the exact proposal artifact bytes. Because
  an adopted clarification changes the pinned proposal bytes, publish a
  successor proposal digest and reconcile the profile hash before source code
  consumes the amended layout.

This amendment does not define the preimages for row-level `input_digest`,
`support_digest`, `transition_digest`, `source_digest`, `config_digest`, or
`checkpoint_digest`; it does not authorize computing those values. Their exact
domains require separate accepted contracts if implementation needs to
produce them. It also does not resolve event generation lifecycle,
same-tick event sequence allocation, tick/time sampling, or canonical score
ledger source.

## Adoption, review, and implementation gate

This file is a proposal, not a decision. It does not change the accepted ABI,
profile digest, schema declarations, allowlist, or implementation authority.
An independent reviewer must review the proposal exact head for arithmetic,
byte ranges, and preserved contract boundaries. Then the named owner and
maintainer must adopt an option in a separate decision artifact. The proposal
and decision require exact-head review; the V4 proposal/reconciliation/profile
pins and transition schema declaration must be republished with the adopted
section order before full codec implementation.

Until that chain is complete, do not implement full bundle packing or reading,
payload digest computation/verification, or complete artifact persistence.
A standalone transition-row-block codec may be separately planned and reviewed
because its field inventory, internal order, and byte size are independently
fixed; it must not claim to produce a V4 bundle or artifact.

No source code, tests, imports, builds, simulator, dashboard, GPU/device,
campaign, lease, checkpoint, or artifact operation was performed to prepare
this proposal.
