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

## Static review boundary

This branch has not run tests, simulators, runtime workloads, or GPU/device
commands. The required review after campaign lease release must verify sensor
RNG semantics, active/inactive row values, initialization, cadence, and seeded
end-to-end replay provenance as specified in the approved plan.
