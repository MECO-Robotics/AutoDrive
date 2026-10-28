# Plan A route evidence readback

The route evidence digest previously copied each of up to twelve tensors from
the accelerator separately. The candidate packs bytes from tensors on one
accelerator into a single transfer, then hashes the existing headers and tensor
bytes in their original order. CPU and mixed-device inputs retain the original
path. The digest schema and replay checks are unchanged.

On a leased WX 9100, eight alternating-order warm pairs used synthetic route
batches with 71 points per path. At 1,000 rows the median was 3.084 ms before
and 2.781 ms after; at 64,000 rows it was 202.369 ms before and 178.154 ms
after. The original and candidate digests matched on every run. These are
component timings, including metadata serialization and SHA-256, not an
end-to-end Plan A training benchmark. The candidate allocates an additional
packed device buffer proportional to the evidence size; its effect on peak
training memory and throughput needs a real A-track capture.

`tests/test_meco_plan_a_route_evidence_digest.py` passed on CPU and `cuda:0`,
including noncontiguous tensor data, mixed dtypes, a missing optional field,
and a mutation check against the original digest implementation. This is a
digest test, not a simulator behavior evaluation. The Plan A HIP dispatch gate
was not opened and no training checkpoint was used.
