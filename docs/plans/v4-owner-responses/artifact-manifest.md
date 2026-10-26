# Artifact/manifest owner response for PPO transition ABI V4

- **Owner role:** `artifact_manifest_owner`
- **Delegated task identity:** `/root/artifact_manifest_owner`
- **Response author:** delegated technical owner `/root/artifact_manifest_owner`
- **Date:** 2026-10-26
- **V4 proposal:** commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`
- **Owner matrix:** commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`
- **Disposition:** **PENDING**

## Scope and evidence

This response covers only the V4 envelope, payload-size arithmetic, byte
encoding, digests, overflow behavior, opaque host persistence, and reader
compatibility in the Artifact/manifest row. It reviews the immutable V4
proposal at the pinned commit and SHA above, as required by the delegated owner
response plan. The owner matrix requests a reproducible field-table sum,
explicit encoding and overflow decisions, and reader compatibility behavior
(`gpu-shared-command-ppo-transition-abi-v4-owner-adoption-matrix-20261026.md`,
Artifact/manifest row).

The proposed 96-byte envelope has a concrete little-endian field layout:
8-byte magic, 16-bit envelope and schema versions, 32-bit header length,
64-bit payload length, 16-bit row count, 8-bit track and reserved fields,
two 32-byte digests, and four reserved bytes (V4 §“Canonical V4 payload
encoding,” header table, lines 95–113). The proposal specifies packed
field-order payload bytes with no implicit alignment, padding, variable-length
columns, or trailing bytes; little-endian integers and IEEE-754 float values;
one-byte 0/1 booleans; and zero values for mask-false floats (same section,
lines 114–121). Those rules make the listed fixed-width layout unambiguous at
the field level.

## Independent byte arithmetic

I independently summed the twelve per-row subtotals in V4’s exact dynamic
row field inventory (lines 140–163):

```text
94 + 30,628 + 512 + 149,959 + 28,728 + 1,983,764
   + 7,628,628 + 99 + 28 + 80 + 122 + 264
= 9,822,906 bytes per row

9,822,906 × 32 rows = 314,332,992 bytes of transition columns
```

For the score-event table, the 13 listed field widths sum to 59 bytes per
record (V4 §“Score-event table and maximum payload,” record table, lines
195–216). At 2,304 records, the table is 135,936 bytes. The bundle-level
identity prefix is 123 bytes: three 32-byte digests, three 64-bit values, one
16-bit code, and one 8-bit mask (V4 lines 122–139). Therefore:

```text
314,332,992  transition columns
    135,936  score-event table (2,304 × 59)
        123  bundle-level identity prefix
------------
314,469,051  maximum payload bytes
         96  header bytes
------------
314,469,147  maximum complete bundle bytes
```

These reproduce the proposal’s arithmetic (V4 lines 226–237). The largest
payload is 314,469,051 bytes, well within the uint64 `payload_bytes` field.
The event count and this maximum are conditional on the proposed fixed
`R=32,S=6,T=12` profile and score-event bound (V4 lines 176–194); they are not
general limits for other profiles.

## Concrete proposal rules and unresolved decision points

The proposal says the payload length is exact for the profile and row count,
the header carries `row_count=32`, and the payload cannot exceed the calculated
maximum (V4 header table, lines 101–112). It recommends reserving full
capacity before mutation and making the complete decision/update a no-op on
overflow, malformed data, unsupported codes, nonfinite valid values, or
digest mismatch, with no truncation or host repair (V4 lines 239–252). It
requires readers to reject unknown versions before parsing and prohibits
converting or reinterpreting V3 bytes (V4 “Authority and decision status,”
lines 33–48). A host may persist/copy the opaque bytes unchanged but may not
decode, hash, compare, select, aggregate, or gate on dynamic payload bytes
(V4 canonical encoding, lines 97–121). These are suitable fail-closed and
opaque-persistence recommendations for this owner row.

I cannot adopt the exact V4 artifact yet because two digest preimages needed
for reproducible artifact validation are not fully specified:

1. `profile_digest` is SHA-256 “of the exact profile/field-layout
   declaration,” but V4 does not define the canonical bytes for that
   declaration: its serialization, domain/version prefix, or precise included
   fields and registries. The digest therefore cannot yet be independently
   reproduced from the proposed declaration (V4 header table, line 110).
2. `payload_digest` is described as SHA-256 of “the exact payload bytes,”
   while the later digest rule says canonical packed bytes with the header’s
   `profile_digest`, row masks, and code registries included (V4 lines
   111 and 294–297). The artifact does not resolve whether the digest input is
   exactly the payload byte interval or a domain-separated preimage that also
   includes header/profile/registry bytes, nor does it define the byte order
   and framing for any such additional inputs. Both interpretations are
   plausible and produce different digests.

To resolve this row, specify immutable, exact digest preimages (including
domain separation, byte framing/order, and whether the profile digest and
code registries are inside or outside the payload interval). Keep the
disposition **PENDING** until those definitions are versioned and the payload
and envelope sizes are checked against the resulting exact field contract.
No replacement artifact is proposed in this response.

## Limits of this decision

This is only the delegated Artifact/manifest technical decision. It does not
decide another owner row, establish cross-owner adoption, define a published
ABI, or authorize implementation. It does not claim current source code
produces this envelope or that the proposed device-side SHA-256 and failure
handling have been implemented or runtime-verified.

This response records only the assigned role’s technical decision. It does
not claim human maintainer approval, cross-owner adoption, ABI publication, or
implementation authorization.
