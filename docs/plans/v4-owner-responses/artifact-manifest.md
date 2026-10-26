# Artifact/manifest owner response for PPO transition ABI V4

- **Owner role:** `artifact_manifest_owner`
- **Delegated task identity:** `/root/artifact_manifest_owner`
- **Response author:** delegated technical owner `/root/artifact_manifest_owner`
- **Date:** 2026-10-26
- **V4 proposal:** commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`
- **Owner matrix:** commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`
- **Disposition:** **PENDING**

This closure response is scoped under the pending-owner-closure plan at commit
`4bef4525acf8022f3b38edaafe0793d13ef7a0a2`, SHA-256
`b0a72285269189f383d25e3463a258fa97a692bdee7674792e213eb0ec8657b7`.

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
general limits for other profiles. The proposal says the producer reserves
the full fixed capacity (lines 240–249), but does not explicitly say whether
every payload serializes all 2,304 event slots or whether it may serialize a
shorter event table. `valid` is present per event and no separate event count
or offset field is listed. The exact `payload_bytes` value therefore still
depends on the serialized-count rule. The Gamepiece/score owner adopted the
59-byte record and 2,304-event batch capacity (response commit
`3738ad58fc883c5853b32c508e89ac657864bca4`, SHA-256
`f57476d13d6038e9e0109fedc557c617ead25a99e4db49512b135a532c1e13af`,
lines 16–20), but that response does not expressly settle whether every slot
is serialized or how a compact count would be framed.

## Concrete proposal rules, source contracts, and closure gates

The proposal defines a 96-byte little-endian header, fixed magic and versions,
`header_bytes=96`, `row_count=32`, zero-valued reserved fields, and the packed
fixed-width encoding above (V4 lines 95–120). It requires a profile mismatch
to reject the bundle before action/optimizer mutation (lines 135–138), and
specifies typed failure with no partial decision/update, truncation, or host
repair for capacity, malformed data, unsupported codes, nonfinite valid
values, and digest errors (lines 240–249). It also says readers reject
unknown versions before parsing and never convert/reinterpret V3 bytes (V4
lines 33–48). These rules establish useful version, byte-order, no-alignment,
opaque-copy, and fail-closed requirements.

The current repository has related but distinct contracts. The campaign
manifest canonicalizes JSON as sorted compact UTF-8 with NaN disallowed and
hashes those bytes (`frc_defense/meco_training_manifest.py:44-50`). It binds
Git/source identity and protocol IDs/hashes, and rejects unsupported schemas
or stale source/protocol bindings (`frc_defense/meco_training_manifest.py:85-88,764-815`).
The current training checkpoint is a `schema_version: 1` Torch payload that
contains config, qualification and evaluation protocol IDs/hashes, source
identity, run ID, model/value/optimizer state, and Torch/Python/device RNG
state, and is written through a temporary file then renamed
(`frc_defense/meco_track_training.py:197-206,2815-2832`). These are evidence
for existing metadata and atomic-write behavior, not a V4 envelope, digest
definition, or profile/checkpoint compatibility contract.

On this follow-up, I expanded the prior PENDING reasons into the following
explicit gates. I cannot adopt exact V4 until the responsible owners decide
them in a versioned, immutable contract:

1. **Digest preimages and framing.** `profile_digest` is SHA-256 “of the exact
   profile/field-layout declaration,” but V4 does not define the declaration's
   canonical byte serialization, domain/version prefix, or exact included
   fields/registries (V4 line 110). `source_digest`, `config_digest`, and
   `checkpoint_digest` are typed as 32-byte values but their preimages and
   linkage to source identity, the profile-bound config, and exact checkpoint
   bytes are not defined (V4 lines 122–138). The campaign manifest's JSON hash
   rule cannot be silently reused for these binary ABI hashes.
2. **Payload digest boundary.** `payload_digest` is SHA-256 of “the exact
   payload bytes,” but the later rule requires canonical packed bytes with the
   header `profile_digest`, row masks, and code registries included (V4 lines
   111 and 294–297). It does not say whether the digest preimage is exactly
   the payload interval or a domain-separated sequence containing header or
   registry bytes, nor specify framing/order for additional inputs. These
   alternatives yield different digest values.
   This is also an open dependency in the PPO/RNG response, which remains
   PENDING and says the per-digest domains/spans for `input_digest`,
   `transition_digest`, `support_digest`, RNG state, and the full batch are
   unspecified (response commit
   `4254f45eca5f2e938000b6e0dda4348ad4ea9337`, SHA-256
   `3f1a05cf9bc4434f413d13d666aa899ad52c4922e46f0390604c87098fa04c10`,
   lines 99–106). The envelope digest rule must be compatible with, and must
   not silently redefine, those row-level digest contracts.
3. **Fixed event-table representation.** The score owner adopted the typed
   event fields and maximum capacity, but the envelope needs an exact decision
   whether every batch serializes 2,304 fixed slots (including invalid slots)
   or includes a count/compact tail. The former gives the computed fixed
   314,469,051-byte payload; the latter needs explicit framing and a revised
   exact length rule. This choice changes `payload_bytes`, the payload digest
   span, and therefore the bundle identity. It depends on Gamepiece/score.
4. **Checkpoint and downstream compatibility binding.** V4 carries source,
   config, checkpoint, and profile digests, but does not specify that the
   checkpoint/sidecar manifest must itself bind the exact V4 profile digest,
   envelope/schema versions, track, protocol identity, and the same source and
   config digest values; nor does it define consumer behavior for a stale or
   missing binding. Current checkpoints and campaign manifests have metadata
   and source/protocol checks, but are separate formats as cited above. The
   exact join keys, bytes to hash, and reject-before-state-mutation rule must
   be specified for producer, checkpoint writer/loader, and downstream
   readers. Evaluation/curriculum owns qualification protocol meaning;
   this row only requires the artifact binding and must not decide that row's
   semantics.
5. **Strict envelope length and malformed-header handling.** The proposal
   supplies header length, payload length, reserved values, magic, row count,
   version rejection, and “no trailing bytes,” but does not give one
   unambiguous reader acceptance predicate for physical buffer length versus
   `header_bytes + payload_bytes`, exact expected payload length (including
   event slots), a wrong `header_bytes`, nonzero reserved bytes, or digest
   verification order. The manifest reader's fail-closed checks show a
   repository precedent for rejecting unsupported schemas and digest/source
   mismatch (`frc_defense/meco_training_manifest.py:764-806`), but do not
   define these binary-envelope cases.

Required closure decision: publish exact digest preimages and envelope reader
acceptance/rejection predicate; bind the same immutable source/config/profile,
checkpoint, track, schema, and protocol identities across the checkpoint
manifest and consumers; and freeze whether the event table is always full
capacity. Recompute/confirm sizes and digest spans after those cross-row
decisions. Until then the disposition remains **PENDING**. No replacement
artifact is proposed in this response.

## Limits of this decision

This is only the delegated Artifact/manifest technical decision. It does not
decide another owner row, establish cross-owner adoption, define a published
ABI, or authorize implementation. Its pending gates depend on Gamepiece/score
for fixed event-table serialization, PPO/RNG for digest coverage and checkpoint
RNG state, Environment/profile for exact profile identity, and
Evaluation/curriculum for protocol identity and qualification-artifact
semantics. It does not claim current source code produces this envelope or
that the proposed device-side SHA-256 and failure handling have been
implemented or runtime-verified.

This response records only the assigned role’s technical decision. It does
not claim human maintainer approval, cross-owner adoption, ABI publication, or
implementation authorization.
