# MECO matched ablation matrix

`frc_defense.meco_ablation_matrix` defines the frozen diagnostic matrix for a
single common five-checkpoint bundle. The runner records each checkpoint file
SHA-256, immutable checkpoint configuration SHA-256, training protocol ID,
source identity, exact category/seed list, condition ID, and evaluation report
identity. Every contrast repeats the common bundle provenance so it can be
audited independently of the full matrix JSON.

Every condition also freezes `qualification_case` with `layout_id`,
`defender_family`, `start_condition`, and `sensor_profile`. These values are
passed to the evaluator and retained in each evaluation report and episode
record. The matrix rejects reports and Phase 4 pairs whose stratum differs from
the frozen condition, so the treatment and no-factor baseline are auditable as
same-category, same-seed, same-stratum comparisons.

## Supported conditions

| Condition | Paired comparison or measurement |
| --- | --- |
| Architecture, CV, K=16 | A1/C2, C1/C2, A1/A2/A3, C1/D1; A1/D0, A2/D0, A3/D0, C1/D0, C2/D0; P=0 versions for A1/A2/A3. C1/D1 both use CV here. |
| Default controls, K=16 | D1=current-pose vs D2=bounded-CV prediction |
| Multi-model controls, K=16 | D1=current-pose vs D2=`multi_model_v1`, with the full predictor contract in control provenance |
| Matched prediction, K=16 | All five learned tracks under current-pose and CV modes |
| Candidate K sweep | C1/C2 candidate recall at K=8, 16, and 32 against the recorded bounded expanded-goal reference pool |
| Inference-disabled fallback | Each learned track paired with its normal run; inference hands off to D1 at the first macro boundary at or after the requested tick |

Evaluation overrides do not rewrite checkpoint configuration. All five
checkpoints must already share the same training configuration and protocol.
The matrix freezes those per-track config hashes and protocol IDs from the
checkpoint files before starting; each evaluation report and each subsequent
condition must match them. Checkpoint and source hashes are rechecked before
and after every condition. Any provenance drift aborts matrix construction
before contrast rows are assembled. The multi-model control condition keeps
the same five-checkpoint bundle, categories, and seed tuple as the default
control condition, changing only D2 prediction mode. Its separate
`D1_vs_D2_multi_model_v1` contrast retains that condition and both control
contracts.
The K sweep changes the runtime candidate capacity for every evaluated policy;
Plan A continues to use its full legal-graph support and does not use K.
P=0 is an inference-time ablation of the same A checkpoint, not a separately
trained checkpoint.

Each learned-track/D0 contrast contains paired outcome rows from
`architecture_matched_cv_k16`, keyed by the exact same category and seed. Rows
retain each record's policy ID and metrics, with condition, evaluation report,
qualification, checkpoint bundle, source, and device provenance. Interpret
these as architecture-condition outcomes: learned tracks use CV, while D0
retains its production deterministic behavior and is not assigned the learned
track CV mode. They are not prediction-mode-matched architecture contrasts.
The matrix harness only assembles and records these comparisons; its output
does not establish that any track wins or has passed simulation qualification.
The multi-model predictor is not calibrated or qualified until held-out paired
calibration and the required seeded simulator evaluation pass.

Example invocation:

```bash
python3 scripts/run_meco_ablation_matrix.py \
  --scenario SCENARIO_ID --categories CATEGORY_A,CATEGORY_B \
  --episodes 20 --seed 20261024 \
  --qualification-protocol qualifications/meco/protocol.json \
  --a1-checkpoint PATH --a2-checkpoint PATH --a3-checkpoint PATH \
  --c1-checkpoint PATH --c2-checkpoint PATH
```

The runner uses the explicit contiguous seed list beginning at `--seed`, or
accepts `--seed-list 11,12,13` for an exact list. For held-out runs, an exact
seed manifest can bind that list to the protocol file hash, scenario,
categories, split, and four-field qualification stratum. Its JSON schema is:

```json
{
  "schema": "meco-exact-seed-manifest-v1",
  "protocol_id": "<protocol id>",
  "protocol_sha256": "<sha256 of frozen protocol file>",
  "scenario_id": "<scenario>",
  "categories": ["<category>"],
  "split": "validation",
  "qualification_case": {
    "layout_id": "<layout>",
    "defender_family": "<family>",
    "start_condition": "<start>",
    "sensor_profile": "<sensor>"
  },
  "seed_list": [11, 12, 13]
}
```

Pass it with `--seed-manifest` and `--qualification-protocol`; every binding
must match and each seed must belong to that exact protocol case. The default
evaluation device is `cuda:0`; CPU is rejected on this accelerator evaluation
path. The runner writes one evaluation JSON per condition plus a matrix JSON.
When `--qualification-protocol` is supplied, every category/seed must be
assigned by that frozen protocol and stratum to exactly one validation or test
split; training, other-stratum, or unassigned seeds are rejected. Each
invocation must resolve entirely to one
split; run validation and test seeds in separate matrices. The matrix passes
the resolved split and protocol ID to every evaluator and records the protocol
ID/hash plus each category/seed assignment. Without that option, the matrix is
explicitly marked `diagnostic_only_unbound_seeds`; it makes no held-out claim.
In either mode these are diagnostic paired records; they do not replace the
calibrated qualification gate and the matrix does not calculate promotion
confidence intervals.

## Optional dashboard replay capture

Replay capture is opt-in and uses the same environment instance and decision
loop that produced each scored evaluation record. Supply both
`--replay-output-dir` and an explicit selection. Legacy `--replay-seeds`
captures those seeds only for the architecture condition. For exact
condition/policy selection, use `--replay-selection-json` with a mapping such
as `{"architecture_matched_cv_k16":{"A1":[11],"D0":[12]}}`; each selected
seed must be in the frozen matrix and only the named policy/condition emits a
replay. This selection is included in the matrix digest.
Point the output directory at a dashboard run's `.scenario-progress`
directory to publish `scenario-meco-*` result, metadata, gzip, compact view,
and ready files. Each captured bundle includes initialization, policy macro
boundaries, and the episode end, plus selected-policy decision annotations.
It binds the exact score-row digest, frame ticks, route/outcome/safety digests,
condition/evaluation IDs, checkpoint/source/protocol/config provenance,
device/backend, qualification stratum, and checkout state. The evaluation
JSON records a pointer and digest for each replay bundle.

For example, capture only A1 on seed 20261024 in the architecture condition:

```bash
python3 scripts/run_meco_ablation_matrix.py \
  --scenario SCENARIO_ID --categories CATEGORY_A,CATEGORY_B \
  --episodes 20 --seed 20261024 \
  --replay-selection-json replay-selection.json \
  --replay-output-dir /path/to/dashboard-run/.scenario-progress \
  --qualification-protocol qualifications/meco/protocol.json \
  --a1-checkpoint PATH --a2-checkpoint PATH --a3-checkpoint PATH \
  --c1-checkpoint PATH --c2-checkpoint PATH
```

Capture copies frame state to the host at each selected episode boundary, so
leave it disabled for ordinary runs. The dashboard replay list marks a bundle
outdated when its source checkout changed or its checkpoint source fingerprint
does not match the evaluation source. This implementation adds capture support;
it does not establish simulator behavior or current dashboard service
verification. Those remain runtime gates after review.

Every final-test matrix invocation reserves one protocol-scoped
`<protocol-id>.sealed-final-test-consumption.json` ledger using exclusive file
creation before simulation starts. It records the exact matrix digest,
protocol ID/hash, source identity, checkpoint hashes, stratum, categories, and
seeds. The same frozen protocol cannot consume a second final-test matrix.
`MecoSupervisor` checks this ledger before starting or resuming training and
rejects training after final-test consumption; a source mismatch is called out
explicitly. Treat a reservation as consumed even if the evaluator crashes,
since the final-test seeds may already have been exposed.

Evaluation records carry per-record wall-time and fallback diagnostics. The
paired summary reports bootstrap differences for end-to-end p50/p95/p99,
route/policy decision pipeline p50/p95/p99, accepted inference p50/p95/p99,
simulator macro p50/p95/p99, and fallback counts/rates. These are explicitly
diagnostic and separate from qualification primary gates. The route/policy
pipeline includes candidate, observation, feature, inference orchestration,
and scoring work; it is not a route-only kernel measurement.

The CLI requires an indexed CUDA/HIP device (default `cuda:0`); CPU and
unindexed `cuda` are rejected. The matrix and each condition ID bind the
normalized accelerator and backend. Each evaluator protocol and report records
the same identity, and matrix validation rejects a report with a different
device or backend.

## Open conditions

The following plan conditions are not represented as completed experiments:

- Phase 4 factors run for registered level 1–5 pressure categories in the two
  pressure scenarios. Level 0 and categories outside that domain remain
  explicitly unsupported and are listed with reasons in the matrix protocol.

They remain listed as open in the matrix protocol. Do not infer that general
noisy or shifted layouts cover these distinct conditions.

## Reviewer checklist

- Confirm all five checkpoint files have the intended common training
  configuration/protocol and inspect recorded file hashes.
- When a qualification protocol is provided, confirm the scenario, categories,
  and per-seed validation/test assignments in `seed_split_assignments`. Without
  one, treat the matrix as diagnostic-only and make no held-out claim.
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
- For `offense_vs_scripted_pressure` or `full_match_offense`, the runner also
  executes the four Phase 4 conditions for requested registered level 1–5
  categories. Each treatment is scoped to those eligible categories and uses
  the exact checkpoint bundle, source identity, and seed list from the
  no-factor condition; all five MECO tracks and D0/D1/D2 controls are
  evaluated. Slot 0 is the treated pressure participant; any other scripted
  pressure slots retain their category behavior. Level 0 has no pressure slot,
  so it remains only in the no-factor run and is recorded as unsupported for
  Phase 4.
- Use the project qualification protocol for promotion; these diagnostic
  records alone do not establish a win or simulator qualification.

## Post-hoc paired confidence intervals

Use `scripts/summarize_meco_ablation_matrix.py` to compute diagnostic paired
95% cluster-bootstrap intervals from one or more already completed matrix
reports. Pass each matrix separately with repeatable `--input-matrix` options:

```bash
python3 scripts/summarize_meco_ablation_matrix.py \
  --input-matrix results/validation_case_v1.matrix.json \
  --input-matrix results/validation_case_v2.matrix.json \
  --bootstrap-seed 20261024 --replicates 10000 \
  --output results/validation_paired_ci.json
```

The analyzer accepts matrix schemas 4 through 8 (new matrices use schema 8) and verifies its content digest,
evaluation report mapping, checkpoint/source/configuration/training protocol
provenance, condition and policy mapping, held-out split, seed coverage, and
each row's frozen qualification stratum. For held-out matrices it checks the
evaluator's protocol-v2 ID and raw-file SHA-256, exact split, scenario,
categories, case stratum, case IDs, and exact episode seed list against the
matrix binding. Diagnostic reports must carry no held-out binding. It records
raw SHA-256 values for every input matrix and referenced evaluation report.
Missing reports or rows, duplicate samples, and bundle or protocol identity
mismatches are rejected. Multiple matrices may be pooled only when
scenario, frozen qualification protocol ID and provenance, checkpoint bundle,
source, training configuration/protocol, device, evaluation design, and split
match. Qualification cases may differ so matched held-out strata can form
distinct bootstrap clusters. Duplicate category/seed/full-stratum samples are
rejected.

Ordinary contrasts use candidate-minus-reference paired metric differences.
The matched prediction contrast uses `cv_metrics - current_metrics`. Candidate
recall is summarized separately per track, category, and K against its declared
bounded reference pool; no recall difference or simulator-optimality claim is
inferred. The bootstrap resamples `(layout_id, defender_family)` clusters,
then paired episode rows within each selected cluster, using the qualification
analyzer's deterministic PRNG and percentile interval method. The seed and
replicate count are recorded in the output.

These intervals are diagnostic summaries. They do not establish
qualification, promotion, causal attribution, or a winning controller. The
qualification gate and its acceptance rules remain authoritative.

For Phase 4, every registered factor must have a paired treatment/no-factor row
for each expected policy and requested category/seed. The analyzer verifies
the factor identity and digest against both the condition manifest and
evaluation report. These intervals remain diagnostic for the evaluated strata.
