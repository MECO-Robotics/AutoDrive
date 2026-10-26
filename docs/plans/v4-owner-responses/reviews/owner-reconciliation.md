# Independent Pass 3 review: V4 owner reconciliation

- **Verdict:** ACCEPT
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** independent exact-head review assigned by `/root`
- **Reviewed reconciliation commit:** `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`
- **Reviewed file:** `docs/plans/gpu-shared-command-ppo-transition-abi-v4-owner-reconciliation-20261026.md`
- **Git blob:** `f160e099155be5f757ae54b15e934524a488479f`
- **Reviewed file SHA-256:** `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`
- **Reconciled profile hash:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`
- **V4 envelope profile_digest:** raw proposal SHA-256 `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.

## Review scope and evidence

Rechecked the exact reconciliation commit, blob, and file SHA-256. The nine ledger entries pin the accepted current response heads and hashes for Environment/profile, Plan A, Plan C, Route, Observation/native, PPO/RNG, Gamepiece/score, Artifact/manifest, and Evaluation/curriculum. I independently verified each response file SHA against its exact commit and each cited persisted review-record SHA against a record that pins that response commit and SHA and records ACCEPT. Earlier REVISE records remain historical; the ledger identifies the current accepted record for each row. All nine current dispositions are ADOPT. The only `PENDING` text is part of the referenced closure-plan filename; no owner row remains pending.

The corrected digest semantics are consistent with the adopted Artifact/manifest decision. That owner response defines the envelope `profile_digest` as SHA-256 of the raw pinned V4 proposal bytes, `0c5ba39...`, and rejects extraction or re-encoding for that field. The reconciliation now preserves that exact value as `v4_profile_digest` and distinguishes it from `reconciled_profile_hash`, which identifies the canonical JSON implementation-plan declaration. I extracted the single-line JSON bytes between the fence markers, excluding a trailing newline, and independently recomputed SHA-256 as `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`, matching the document. This resolves the digest conflict from the superseded REVISE review.

Independently recomputed the packed field totals from the pinned V4 inventory. Plan A `support_valid` is the 96,768-element boolean support vector. The per-row groups sum to 9,822,906 bytes; 32 rows use 314,332,992 bytes. The score table is 2,304 × 59 = 135,936 bytes; static identity is 123 bytes; payload is 314,469,051 bytes; adding the 96-byte header gives 314,469,147 bytes. The corrected size conversion is 299.9011 MiB at four decimal places. These values match the JSON and byte derivation. The N/S/R/T/E/K, Plan A/Plan C capacities, route sink/chunk/path limits, observation handles, score-event table, and envelope limits are compatible with the accepted owner rows and V4 field inventory.

The reconciliation accurately identifies remaining source implementation and qualification work, distinguishes its proposal status from ABI publication and human approval, and makes no claim of source conformance, GPU-only compliance, training results, or Stage 1 qualification. Its next-step language keeps implementation behind a separate plan and review.

## Review limits

This is an independent technical review of this exact reconciliation artifact only. It does not publish V4 as normative, authorize implementation, or establish qualification or governance approval. No tests, imports, compilation, runtime/device, simulator, dashboard, checkpoint, campaign, training, or evaluation actions were performed.
