# MECO validation campaign planner

`scripts/plan_meco_validation_campaign.py` compiles the exact validation cases in an all-scenario frozen qualification protocol. It does not invoke the simulator, start GPU work, consume final-test seeds, or run a matched matrix.

The planner groups categories only when their validation seed lists are byte-for-byte equal for the same scenario and stratum. It emits a deterministic `plan.json` and one protocol-bound seed manifest per scenario/stratum. The plan records each default matrix condition, expected policy × category × seed coverage as explicit axes plus a digest of the expanded keys, and a small replay selector for the first validation seed under the architecture-matched condition. Optional inputs bind all five checkpoint hashes, a serialized `source_identity()` object, and five training-configuration hashes.

The plan reports whether its protocol is qualification-eligible. A provisional protocol may produce a planning artifact for diagnostic preparation, but `--qualification-claim` rejects it. Qualification claims require a calibrated v2 protocol with frozen safety gates. The planner verifies the protocol digest, registered-scenario coverage, case completeness, split disjointness, and seed ownership before writing artifacts. Final-test seed lists are never copied into the plan or seed manifests.

Example:

```bash
python3 scripts/plan_meco_validation_campaign.py \
  --protocol qualifications/meco/protocol.json \
  --output-dir evaluations/meco-validation-campaign
```

To bind the current source and checkpoint bundle, also supply `--source-identity-json`, `--training-configuration-hashes-json`, and all five `--a1-checkpoint` through `--c2-checkpoint` options. Re-running with identical inputs is idempotent; conflicting existing artifacts are rejected.
