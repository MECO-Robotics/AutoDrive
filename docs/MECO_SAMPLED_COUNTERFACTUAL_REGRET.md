# Sampled counterfactual regret

`scripts/meco_counterfactual_regret.py` analyzes a C1 or C2 checkpoint against
strictly replay-bound `meco-counterfactual-trajectory-v3` data. The analyzer
regenerates the archived route-feasible candidate table, verifies every
counterfactual descriptor and outcome through the existing v2 replay validator,
and scores the exact table with the checkpoint's masked utility minus the
replayed route and prediction-risk costs.

For a branch point with an outcome for the checkpoint's exact selected
candidate ID, the reported quantity is:

```text
max(gross_horizon_return over sampled available candidate outcomes)
- gross_horizon_return of the checkpoint-selected candidate
```

This is **sampled bounded-set regret**. Counterfactual-v2 stores outcomes for a
deterministic bounded sample of valid candidates, so this value does not bound
regret against unsampled candidates, the full action space, or a globally
optimal simulator policy. If the checkpoint selects a candidate whose outcome
was not sampled, that branch point is excluded from the regret aggregate and
counted in the chosen-outcome coverage report. Candidate IDs must match exactly;
there is no nearest-row substitution. Exact score ties select the first valid
row in the archived stable candidate-table order.

The analyzer binds the data and manifest byte hashes, source identity,
qualification protocol, prediction mode and contract, checkpoint byte hash,
checkpoint inference configuration hash, candidate boundary, per-row candidate
table and sampled-support digests, and scoring semantics in the output JSON.
For each branch point, the analyzer scores every exact archived teacher-prefix
candidate table in order, applies the checkpoint's tracking-loss hidden-state
reset rule, and carries the C1/C2 GRU state into the branch decision. It also
reexecutes every sampled descriptor and compares gross macro/horizon returns,
discounted return, continuation count/policy, discount semantics, and terminal
status with the archived row before computing regret. These checks replay the
simulator repeatedly and can make analysis slow; no archived return is used
unless strict replay reproduces it.

D1 requires current-pose risk and D2 requires bounded constant-velocity risk.
The v2 source artifact contains one prediction-mode risk tensor per archived
table, and this analyzer does not independently execute either control scorer.
It marks both controls unavailable and does not treat two uses of one risk
tensor as independent D1 and D2 comparisons. The primary report is the C
checkpoint's sampled-set regret.

Run with an explicit indexed accelerator matching the dataset/checkpoint
configuration:

```bash
python scripts/meco_counterfactual_regret.py \
  --track C1 \
  --dataset evaluations/meco-supervised/counterfactual-v2.jsonl \
  --checkpoint checkpoints/meco-five/C1/latest.pt \
  --qualification-protocol evaluations/meco-qualification/protocol.json \
  --device cuda:0 \
  --output evaluations/meco-supervised/C1-sampled-regret.json
```

The analyzer performs strict simulator replay when run. Static implementation
checks do not establish replay parity, runtime scoring correctness, or
simulator behavior; produce and inspect a fresh simulator evaluation/replay
before treating any generated report as runtime-validated evidence.
