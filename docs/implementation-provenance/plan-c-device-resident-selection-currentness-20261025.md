# Plan C dashboard currentness evidence — pending

Status: **not verified**. This note is a pending checklist for the post-merge
currentness stage in the approved plan. No dashboard service was restarted or
queried for this D candidate. Do not treat the API digest pin or local
provenance tests as evidence that a running dashboard serves this checkout.

After E is approved and integrated, record the UTC timestamp, active service
PID/invocation, working directory, integrated source identity, HTTP results for
`/api/curriculum-status` and `/api/meco-status`, and all eight frontend route
hash comparisons specified in the approved plan. The curriculum response must
expose `three_pass_provenance.validation_status == "validated_summary_only"`,
manifest `status == "scoped-transcript-summaries"`, 89 rows, the exact final
requirement ID, and no `untrusted_reason`. Separately compare the local ledger
SHA-256 to `_EXPECTED_THREE_PASS_PROVENANCE_SHA256` in the same checkout.

Until those checks are completed and recorded, dashboard currentness remains
pending. This tranche supplies no simulator, replay, GPU/HIP, training, or
qualification evidence.
