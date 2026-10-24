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

## Validation-only campaign execution

`scripts/run_meco_validation_campaign.py` compiles the immutable plan from the raw
protocol, five checkpoints, checkpoint training configurations, and the prepared
source identity, then executes all 28 jobs sequentially on one explicit indexed
accelerator. It rejects a qualification claim, requires the current provisional
protocol, and never exports final-test seeds. Each job gets a separate output
directory and an immutable seed manifest, input record, evaluation reports,
replay exports, and completion seal.

```bash
python3 scripts/run_meco_validation_campaign.py \
  --protocol qualifications/meco/protocol.json \
  --output-dir evaluations/meco-validation-campaign \
  --device cuda:1 \
  --a1-checkpoint checkpoints/A1.pt \
  --a2-checkpoint checkpoints/A2.pt \
  --a3-checkpoint checkpoints/A3.pt \
  --c1-checkpoint checkpoints/C1.pt \
  --c2-checkpoint checkpoints/C2.pt
```

The runner holds an exclusive host-wide device lock and an output-campaign lock.
It does not clear stale locks. Resume accepts only a valid completion seal or a
fully validated matrix missing its final seal; partial condition or replay
artifacts fail closed for manual inspection. After the 28-job campaign seal is
written, verify and summarize it with:

```bash
python3 scripts/summarize_meco_validation_campaign.py \
  --plan evaluations/meco-validation-campaign/plan.json \
  --campaign evaluations/meco-validation-campaign/campaign.complete.json \
  --output evaluations/meco-validation-campaign/summary.json
```

The summary keeps jobs nested under their scenario and leaves the pooled estimate
unset because no operational scenario weights have been specified. These
commands execute simulator evaluations when invoked; contract tests use a mocked
matrix runner and do not execute simulations.
