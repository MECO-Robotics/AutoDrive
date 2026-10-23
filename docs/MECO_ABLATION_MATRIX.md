# MECO matched ablation matrix

`frc_defense.meco_ablation_matrix` defines the frozen diagnostic matrix for a
single common five-checkpoint bundle. The runner records each checkpoint file
SHA-256, immutable checkpoint configuration SHA-256, training protocol ID,
source identity, exact category/seed list, condition ID, and evaluation report
identity. Every contrast repeats the common bundle provenance so it can be
audited independently of the full matrix JSON.

## Supported conditions

| Condition | Paired comparison or measurement |
| --- | --- |
| Architecture, CV, K=16 | A1/C2, C1/C2, A1/A2/A3, C1/D1; P=0 versions for A1/A2/A3. C1/D1 both use CV here. |
| Default controls, K=16 | D1=current-pose vs D2=bounded-CV prediction |
| Matched prediction, K=16 | All five learned tracks under current-pose and CV modes |
| Candidate K sweep | C1/C2 candidate recall at K=8, 16, and 32 against the recorded bounded expanded-goal reference pool |
| Inference-disabled fallback | Each learned track paired with its normal run; inference hands off to D1 at the first macro boundary at or after the requested tick |

Evaluation overrides do not rewrite checkpoint configuration. All five
checkpoints must already share the same training configuration and protocol.
The matrix freezes those per-track config hashes and protocol IDs from the
checkpoint files before starting; each evaluation report and each subsequent
condition must match them. Checkpoint and source hashes are rechecked before
and after every condition. Any provenance drift aborts matrix construction
before contrast rows are assembled.
The K sweep changes the runtime candidate capacity for every evaluated policy;
Plan A continues to use its full legal-graph support and does not use K.
P=0 is an inference-time ablation of the same A checkpoint, not a separately
trained checkpoint.

Example invocation:

```bash
python3 scripts/run_meco_ablation_matrix.py \
  --scenario SCENARIO_ID --categories CATEGORY_A,CATEGORY_B \
  --episodes 20 --seed 20261024 \
  --a1-checkpoint PATH --a2-checkpoint PATH --a3-checkpoint PATH \
  --c1-checkpoint PATH --c2-checkpoint PATH
```

The runner uses the explicit contiguous seed list beginning at `--seed` and
writes one evaluation JSON per condition plus a matrix JSON. These are
diagnostic paired records; they do not replace the calibrated held-out
qualification gate and the matrix does not calculate promotion confidence
intervals.

## Open conditions

The following plan conditions are not represented as completed experiments:

- Phase 4 factors remain unsupported for categories outside the registered
  reactive-pressure domain; see the explicit scenario constraints below.

They remain listed as open in the matrix protocol. Do not infer that general
noisy or shifted layouts cover these distinct conditions.

## Reviewer checklist

- Confirm all five checkpoint files have the intended common training
  configuration/protocol and inspect recorded file hashes.
- Confirm the exact scenario, categories, and seed list are the held-out split
  approved for this diagnostic.
- Inspect `conditions` and verify all condition IDs include the same checkpoint
  bundle, source identity, and seeds.
- Inspect each contrast's `common_bundle_provenance` and
  `report_provenance`; ensure both sides point to the intended condition and
  evaluation reports.
- Check that D1/D2 uses current-pose/CV in the default-control condition, and
  that the all-five prediction comparison uses matched current-pose and CV
  conditions.
- Check K=8/16/32 recall rows and interpret the reference strictly as the
  declared bounded route-qualified pool.
- Check fault records for requested tick, actual trigger tick, `reached`,
  fallback count, and paired normal policy ID.
- For `offense_vs_scripted_pressure` or `full_match_offense` with only registered
  `level_4_reactive` categories, the runner also executes the four Phase 4
  conditions from `docs/MECO_PHASE4_FACTORS.md`. Each treatment uses the exact
  checkpoint bundle, source identity, categories, and seed list from the
  matched no-factor condition; all five MECO tracks and D0/D1/D2 controls are
  evaluated. Unsupported categories leave these conditions explicitly open.
- Use the project qualification protocol for promotion; these diagnostic
  records alone do not establish a win or simulator qualification.
