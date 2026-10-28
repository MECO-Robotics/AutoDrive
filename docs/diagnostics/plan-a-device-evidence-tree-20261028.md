# Plan A route evidence tree digest prototype

Route evidence binds source identity, query ordering and metadata, and each
route, route-cost and risk-cost tensor byte. The existing `sha256-v1` format
hashes that canonical byte stream directly. Its GPU route copies the packed
stream to the host for hashing.

The prototype `tree-sha256-v2` hashes the same byte stream in fixed 1,024-byte
chunks, recursively hashes ordered chunks of those hashes until at most 1,024
bytes remain, then hashes a domain tag, the big-endian stream length and chunk
size, and those remaining hashes. It is a different digest: rows record
the scheme, and GPU-only PPO requires v2 before any optimizer step. Legacy
v1 fingerprints remain available for offline inspection but are not accepted
for training replay. Mixed-device v2
inputs fail closed. CPU digest support provides a reference oracle.

The HIP path requires a separately compiled native module. Its maintained
`.cpp` and `.cu` sources are included in source identity; its generated HIP
and C++ translations are excluded from source identity so compilation cannot invalidate a
running campaign. Neither the route shortest-path sources nor production route
gates are changed. Checkpoint source identity still changes and existing
checkpoints require their established migration or warm-start procedure.

This is a prototype pending representative replay and actual A-track rollout
validation. Its report is not a training speedup.

The recursive prototype passed 30 focused CPU and HIP checks, including digest
parity around SHA padding and multi-level chunk boundaries. A paired synthetic
12,096-query, 73-point route batch measured median 32.57 ms for v1 and
21.14 ms for v2 over five alternating pairs. This includes query serialization,
packing, hashing and digest readback; it excludes route generation and training.
The GPU result matched the CPU tree oracle. Measurements and source hashes are
retained at `autodrive-plan-a-device-evidence-20261028-evidence/route-digest-pair-hier.json`.
A fresh simulator replay and actual Plan A rollout remain required.
