# GPU observation and sensor preprocessing implementation tranche

This tranche implements Pass 2 against approved plan commit
`dd38fa507ae7840d65854333fad2595a41460b7d`, artifact SHA
`df54d318ebed2d22946196a80bf4baa5a5fb61a2eb4899a6ccdcb0a2c4de2e5f`.
The reviewed source base is `53f3d9c9a49a7e12bf65b5e1fa2797f4464e63be`.

## Source contract changes

- Observation noise, dropout, and sensor draws allocate fixed `[N, ...]` device
  batches and apply active masks with tensor selection. They no longer compact
  from a host count or call `active.sum().item()`.
- This changes RNG consumption for partial active batches: draws now include
  inactive rows. Seed parity is not claimed. Runtime/observation identity and
  seeded fixtures must be regenerated and reviewed after device access resumes.
- Environment step methods no longer derive active counts from device masks or
  return early based on those counts. Optional counts are accepted only as
  caller-known scheduler metadata. Partial-batch state is preserved by masked
  tensor updates.
- Held-action scheduling requires the caller's scheduler-known remaining-tick
  scalar. The runtime no longer copies `env.steps` to CPU or reduces it there
  to synthesize per-tick active counts.
- Defense reset calls disable the scalar planner cadence fast path without
  inspecting `mask.all()` on the host. Per-world cadence remains authoritative.
- 3v3 resets follow the same per-world cadence rule; a device reset mask never
  restores its scalar full-batch counter. Tensor masks and returned observation
  batches receive static device/shape guards at perception and policy boundaries.
- GPU paths reject CPU or wrong-device active/reset masks at their observation
  boundaries. The NumPy normalizer remains defined for offline adapters; source
  audit found no call to it from the simulator observation callchains. Tensor
  normalization now checks that observation and scale tensors share a device
  and that the feature width supports the fixed 35-value scale schema.
- Strategic evaluation active counts are maintained by its host episode quota
  scheduler while processing already-finished episodes; reset cardinality is no
  longer reduced from `reset_mask` to the host.

## Static review boundary

This branch has not run tests, simulators, runtime workloads, or GPU/device
commands. The required review after campaign lease release must verify sensor
RNG semantics, active/inactive row values, initialization, cadence, and seeded
end-to-end replay provenance as specified in the approved plan.

## Approved plan extension and follow-on changes

The subsequent Pass 1-approved plan is `2d67e108da7dd99ddf353cd5d5d25261c48abc06`,
artifact SHA `4517ca3fcf0970dc0466287e3889d7e9b3de4a47fa9999647b30f79ee4b0e5c0`.
It extends this tranche to active-derived planner/reset predicates and the
`TensorSim.step` active-mask contract. `TensorSim.reset_done` and randomized
physics parameters now draw fixed `[N, ...]` batches and mask writes, changing
RNG consumption for inactive rows and partial resets; seeded parity is not
claimed. Empty active masks proceed through fixed-shape masked physics rather
than a host early return. These changes have static checks only and require
runtime review after campaign access is released.

Planner cadence, focal traversal overrides, depot retries, and optional heading
alignment now use device masks and fixed-shape planner calls. Macro execution
propagates target-valid/action-match masks to later ticks; tensor-to-host
projection happens after execution for trace/report fields. This removes the
early host break on those status masks and may run masked no-op ticks through
the requested macro horizon. Runtime parity and throughput are unverified.

All 3v3 planner calls set `static_active_mask=True`, avoiding `TensorADStar`
caller-mask `nonzero` compaction. The Pass 3 tranche also removes host
predicates from endpoint projection and route extraction, validates GPU
learned-cost maps with device assertions, and uses fixed-shape device draws for
defender placement. These changes alter random-number consumption for inactive
reset rows; seeded parity is not claimed. GPU local path repair now fails closed
to the existing one-point stop route; the CPU local repair path remains intact.
Route extraction runs its fixed iteration count rather than stopping early
from a host reduction. Static checks only; runtime parity and throughput remain
unverified.

Reset-time defender planning also uses a static full-shape active mask, and the
focal pose-estimate hook validates finite values with the device assertion
helper because it writes directly into policy-visible pose state.
