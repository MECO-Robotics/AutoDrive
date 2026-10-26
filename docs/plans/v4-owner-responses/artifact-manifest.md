# Artifact/manifest owner response for PPO transition ABI V4

- **Owner role:** `artifact_manifest_owner`
- **Delegated task identity:** `/root/artifact_manifest_owner`
- **Response author:** delegated technical owner `/root/artifact_manifest_owner`
- **Date:** 2026-10-26
- **V4 proposal:** commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`
- **Owner matrix:** commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`
- **Closure-plan input:** commit `4bef4525acf8022f3b38edaafe0793d13ef7a0a2`, SHA-256 `b0a72285269189f383d25e3463a258fa97a692bdee7674792e213eb0ec8657b7`
- **Disposition:** **ADOPT** the exact pinned V4 proposal for the Artifact/manifest owner row, with the concrete artifact interpretations below.

This response is limited to the Artifact/manifest owner row. Its decision is a
delegated technical decision, not human maintainer approval, cross-owner
adoption, shared ABI publication, or implementation authorization.

## Scope and evidence

The V4 proposal defines a 96-byte little-endian header, fixed magic and
versions, `header_bytes=96`, `row_count=32`, zero-valued reserved fields, and a
packed fixed-width payload with no implicit alignment, padding, variable-length
columns, or trailing bytes (V4 §“Canonical V4 payload encoding,” lines 95–120).
Integers and floats are little-endian; booleans are one byte; invalid masked
floats are zero. Profile mismatch must reject before action or optimizer
mutation (V4 lines 135–138). Unknown versions are rejected before payload
parsing, and V3 bytes are not converted or reinterpreted (V4 lines 33–48).

## Independent byte arithmetic and fixed length decision

I independently summed the twelve per-row subtotals in V4’s exact dynamic
row field inventory (lines 140–163):

```text
94 + 30,628 + 512 + 149,959 + 28,728 + 1,983,764
   + 7,628,628 + 99 + 28 + 80 + 122 + 264
= 9,822,906 bytes per row

9,822,906 × 32 rows = 314,332,992 transition-column bytes
```

The 13 score-event fields total 59 bytes per record (V4 §“Score-event table
and maximum payload,” lines 195–212). The adopted capacity is 2,304 records,
so the event table is `2,304 × 59 = 135,936` bytes. The static identity prefix
is 123 bytes (three 32-byte digests, three 64-bit values, one 16-bit code, one
8-bit mask; V4 lines 122–139). Thus:

```text
314,332,992  transition columns
    135,936  fixed score-event table
        123  bundle-level identity prefix
------------
314,469,051  exact payload bytes
         96  header bytes
------------
314,469,147  exact complete bundle bytes
```

These reproduce V4’s arithmetic (lines 226–237). I decide that every valid
bundle serializes all 2,304 event slots. Unused slots have `valid=0` and all
other fields zero (float fields use positive zero); there is no event count or
compact tail. This is consistent with V4’s fixed record/capacity and full
capacity reservation requirement (lines 195–212, 240–249), and with the
Gamepiece/score owner’s adoption of the 59-byte record, 2,304-event capacity,
and reservation of the full table before mutation (response commit
`3738ad58fc883c5853b32c508e89ac657864bca4`, SHA-256
`f57476d13d6038e9e0109fedc557c617ead25a99e4db49512b135a532c1e13af`,
lines 16–20 and 72–78). If capacity is exceeded or a record marked valid has
invalid fields or codes, the whole staged decision/update fails; no truncated
or partial bundle is published. The exact length is profile-specific and must
be recomputed for any versioned replacement that changes a field, dimension,
bound, or encoding. The PPO digest-span decision and Gamepiece/score table
decision add no fields or byte widths, so the reproduced V4 sizes remain
unchanged.

## Adopted identity and digest bindings

The following choices make the V4 digest fields reproducible without changing
their types or the payload layout:

- **`profile_digest`:** the 32 raw SHA-256 bytes of the exact Git blob for the
  pinned V4 proposal file. For this proposal the value is
  `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`. Hash
  the raw artifact bytes without Markdown normalization, extraction, or
  re-encoding. The pinned file is the immutable profile/field-layout
  declaration for this digest; this conservatively binds its exact artifact
  bytes, not a normalized or extracted table. Any byte change requires a new
  immutable artifact and hash; bytes from different hashes do not match this
  profile.
- **`source_digest`:** the raw 32-byte value of the matching checkpoint/run
  manifest’s `source_identity_sha256`. Its preimage is the exact `sources`
  object serialized as
  `json.dumps(sources, sort_keys=True, allow_nan=False).encode("utf-8")`,
  matching the trainer at `frc_defense/meco_track_training.py:2266–2268`.
- **`config_digest`:** the raw 32-byte `configuration_sha256` for the track
  in the matching run manifest. Its preimage is the checkpoint `config`
  object with `updates` omitted, serialized with the same sorted-key,
  `allow_nan=False` JSON rule (`frc_defense/meco_track_training.py:2332–2339,
  2379–2381`). This matches the current resume-compatibility rule for the
  per-invocation update target.
- **`checkpoint_digest`:** SHA-256 of the exact checkpoint file bytes, with no
  deserialization or re-encoding; this is the operation in
  `frc_defense/meco_training_manifest.py:81–82`. An immutable sidecar manifest
  stores this result; the checkpoint file does not contain its own digest.
  Each manifest names that exact checkpoint generation and hash. A mutable
  `latest` pointer is only a locator; the loader re-hashes the resolved bytes
  and rejects if they do not match the immutable manifest.
- **`payload_digest`:** SHA-256 over exactly the fixed payload byte interval,
  excluding the 96-byte header and external framing. The adopted PPO/RNG
  response explicitly says the bundle payload digest covers the exact full
  payload bytes and defines the separate row-digest domains/spans (response
  commit `602565ef99986996d2c12d06024833247dc3d395`, SHA-256
  `219f3466741c17d9e49a4d402f0b3f8ae6f016f0738a068034a8ad312f610d75`,
  §“Philox streams, checkpoint continuation, and digest bytes,” item 5). The
  bundle digest does not replace or redefine `input_digest`, `support_digest`,
  or `transition_digest`.

The V4 checkpoint and its immutable manifest bind the same profile, source,
configuration, checkpoint, track, envelope/schema, and training/qualification
protocol identities. The checkpoint continuation state also binds root seed,
action/permutation stream codes and counters, RNG protocol/source/profile
identity, and model/optimizer state as one continuation point, following the
adopted PPO/RNG response (same commit, item 4). A loader rejects missing,
malformed, unsupported, or mismatched bindings before restoring model,
optimizer, or RNG state. The checkpoint manifest’s checkpoint digest is a
sidecar reference to the exact checkpoint bytes, avoiding self-hashing. A V1
Torch checkpoint cannot be resumed as V4. Downstream evaluation manifests
bind the exact checkpoint-file digest and their protocol ID/hash; this row
does not decide evaluation protocol meaning or qualification gates.

## Reader, persistence, and failure behavior

For this pinned V4 profile, a reader:

1. Requires at least the 12-byte common prefix to check magic and the two
   version fields. It rejects bad magic or unsupported envelope/schema
   versions before payload parsing.
2. For `envelope_version=2` and `schema_version=4`, requires a 96-byte header,
   32 rows, track code `1..5`, zero reserved fields, the exact profile digest
   above, and `payload_bytes` exactly `314,469,051`. It uses checked length
   arithmetic and requires physical
   bundle length exactly `header_bytes + payload_bytes`; truncated, oversized,
   or trailing-byte buffers fail.
3. Verifies the payload digest over the entire fixed payload interval, then
   validates static identities, dimensions, code registries, row masks,
   finite-value rules, and score-event slots on device before any action,
   optimizer, or RNG state mutation. Any failure rejects the whole bundle.

The host may persist and copy the opaque bundle unchanged, but does not decode,
hash, compare, select, aggregate, or gate on dynamic payload bytes (V4 lines
97–120). No V3 conversion or partial artifact is allowed. Failure behavior is
fail-closed with no state mutation, truncation, or host repair (V4 lines
240–249).

## Current-source boundary

The repository has related but distinct legacy contracts. Campaign-manifest
JSON hashes use sorted compact UTF-8 JSON with NaN disallowed
(`frc_defense/meco_training_manifest.py:44–50`); manifest validation rejects
unsupported schemas and stale source/protocol bindings (lines 764–815). The
current trainer’s checkpoint is schema version 1 and contains config,
qualification/training protocol identifiers, sources, model/value/optimizer
state, and Torch/Python/device RNG state; it is written to a temporary file
then renamed (`frc_defense/meco_track_training.py:197–206,2815–2832`). These
source facts do not mean V4’s binary envelope, digests, checkpoint identity, or
reader behavior are implemented. V4 requires a new versioned continuation
binding; the legacy schema is not a compatible substitute.

This ADOPT records only the delegated Artifact/manifest decision for the
pinned V4 proposal. It does not decide other owner rows, establish cross-owner
adoption or shared ABI publication, claim implementation/runtime evidence, or
authorize source changes. The response is not human maintainer approval.
