# Independent Pass 3 review: V4 owner reconciliation

- **Verdict:** REVISE
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** independent exact-head review assigned by `/root`
- **Reviewed reconciliation commit:** `78f5adba67de246fad20883b9560e3d5f4640516`
- **Reviewed file:** `docs/plans/gpu-shared-command-ppo-transition-abi-v4-owner-reconciliation-20261026.md`
- **Git blob:** `6166b1259985916e07325b60bb92880461d265b0`
- **Reviewed file SHA-256:** `7b5d07afd91245af07071220402a2a7f31f5e41b394142b9d0c8c539c01d81d3`
- **Declared canonical profile hash:** `25c03d281f8866714bc8b58308373de827c0e6fb926d5f498497c9af685cb476`
- **Pinned V4 proposal:** commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
- **Pinned owner matrix:** commit `ee234b2c5162038e6fcfff6d9551d80f443f6d82`, SHA-256 `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53`.

## Review scope and evidence

Rechecked the exact reconciliation commit, blob, and file SHA-256. I verified the nine ledger response file hashes against their pinned response commits and verified each cited review-record file hash against a persisted record containing an ACCEPT for that exact response commit and response SHA. The Environment/profile, Plan A, Plan C, Route, Observation/native, PPO/RNG, Gamepiece/score, Artifact/manifest, and Evaluation/curriculum rows are each ADOPT in the ledger. The artifact contains no owner disposition left PENDING; the one PENDING filename mentioned is the source closure-plan path. The response/review pins and the capacity declarations are otherwise coherent with the reviewed owner records.

The canonical JSON bytes inside the fenced block, with no newline, hash to `25c03d281f8866714bc8b58308373de827c0e6fb926d5f498497c9af685cb476`, matching the document's computed declaration hash. Independent byte recomputation also matches the declared row and bundle byte counts. Interpreting `support_valid` as the 96,768-element Plan A support vector (the adopted `A_G=96,768` capacity), the per-transition groups sum to 9,822,906 bytes; ×32 gives 314,332,992 bytes. The 2,304 × 59-byte event table is 135,936 bytes, the static prefix is 123 bytes, payload is 314,469,051 bytes, and the 96-byte header gives 314,469,147 total bundle bytes. The displayed MiB conversion is slightly inaccurate: 314,469,147 bytes is 299.9011486 MiB, which rounds to 299.9011 MiB at four decimal places, rather than 299.9012 MiB.

There is a blocking cross-owner digest conflict. The accepted Artifact/manifest response at commit `45dc7b4894d97fd7aa04b1f84aac3cdc207f0c9d` (file SHA-256 `9b7a571bd98c6aed6700d445aea8e05a2308cdc5c1bcba9c93c961c4194d8289`) adopts `profile_digest` as the SHA-256 of the raw bytes of the exact pinned V4 proposal file, value `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`, and explicitly says not to extract, normalize, or re-encode that artifact. This reconciliation instead defines the SHA-256 of an extracted JSON declaration, `25c03d...`, as the V4 `profile_digest` and final reconciled profile hash. Those values differ. This changes an adopted digest-field contract and means the nine accepted rows are not yet compatible on the profile identity, despite the reconciliation's all-compatible status. Resolve the conflict through the authorized owner/reconciliation process and obtain any required exact-head review before treating this artifact as accepted.

The byte-length arithmetic itself remains correct; the digest conflict affects the canonical identity value, not the byte count. Apart from this blocker and the MiB rounding correction, the capacity relationships and the explicit limits against implementation, ABI publication, governance approval, source conformance, GPU-only compliance, and qualification are accurately stated. This review does not authorize implementation or ABI publication.

## Review limits

This verdict is an independent technical review of this exact reconciliation artifact only. No tests, imports, compilation, runtime/device, simulator, dashboard, checkpoint, campaign, training, or evaluation actions were performed. It is not human approval, ABI publication, implementation authorization, or qualification.
