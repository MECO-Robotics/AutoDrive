# MECO evaluation protocol binding

Evaluation split identity is separate from the supervisor's training
qualification binding. Without an evaluation protocol, API and supervisor
evaluations are diagnostic and carry no held-out qualification claim. An
explicit `validation` or `test` split requires a protocol whose content ID and
file SHA are rechecked by the evaluator before device setup or simulator work.

The evaluator requires an exact case stratum (`layout_id`, `defender_family`,
`start_condition`, `sensor_profile`) and the exact frozen seed list for every
requested category. Missing or ambiguous cases, altered protocol content,
seed-list mismatches, and cross-split seed leakage fail closed. The resulting
evaluation ID, protocol ID and SHA, split, case, and exact seeds appear in the
evaluation protocol hash, status sidecar, report, and each record. Training
qualification IDs remain in their own fields.

The continuous supervisor supports only diagnostic evaluations by default or
explicit bound validation evaluations. It never auto-selects final test. When
the requested categories map to multiple validation strata, select one with
`--evaluation-case-json`:

```bash
python3 -m frc_defense.meco_supervisor \
  --devices cuda:0,cuda:1 --evaluation-device cuda:1 \
  --evaluation-protocol qualifications/meco/protocol.json \
  --evaluation-split validation \
  --evaluation-case-json '{"layout_id":"2026_rebuilt_shifted_structures_v1","defender_family":"none","start_condition":"offset_upper_lane","sensor_profile":"noisy"}'
```

The supervisor resolves one shared exact seed list across its categories and
passes it with the protocol path and case to the evaluator. Leave off the
evaluation protocol and split options for unbound diagnostic evaluation.
Final test remains an explicitly invoked offline evaluator operation, with its
own protocol case and exact seeds.

These changes have CPU contract coverage only. Per project `AGENTS.md`, any
runtime behavior evidence or promotion still requires a seeded end-to-end
simulator evaluation, inspection of its record/replay, and a restarted and
verified current dashboard backend and frontend with a fresh replay. That
runtime gate was not run as part of this implementation.
