# Architecture study runbook

> Completed V5 study: all 119 training cases passed the structural/checkpoint audit; all 15 CNN map diagnostics, six absence references, and four fresh replay reviews are complete. See [findings](NN_ARCHITECTURE_FINDINGS.md) for results, the known baseline test failure, and trajectory reproducibility limits. Earlier comparisons remain withdrawn.

Run from `/home/brian/AutoDrive` with its `.venv`. The default architecture list includes all seven alternatives: baseline MLP, residual MLP, Deep Sets, entity Transformer (`entity_attention`), full-field CNN, full-field ResNet, and field FNO. These commands document the short screen; they do not establish an architecture ranking or convergence.

The corrected rerun uses frozen source snapshot **v5**, hash `bfa926060795a3c97422ab15f1cee94a15cd31ca9ce84c79cee1e1cd0a5be32f`, archived in `evaluations/architecture-study-20261020/study-source-v5.tar.gz`. All 119 v4 cases are retained under `legacy-planner-truth-v4/matrix`: their structural/checkpoint audit passed, but planner obstacles, proactive avoidance and contact-track invalidation used hidden enemy truth. Those results are not strict perception evidence. V5 corrects the complete observation-to-command boundary, enables simulated opponent sensing for neural offense, and removes a callback lifetime cycle that could defer GPU graph destruction into later environment construction. Every case is rerun on the same source; no pickup cases are reused.

Earlier v1/v2/v3 results were independently withdrawn because cached GPU target graphs retained replaced scoring-grid tensors across resets. The fix invalidates that graph on grid rebuild, preserves fallback perception-age storage, and prevents HIP graph capture on CPU. Raw invalid runs remain under `outdated-nondeterministic-reset/OUTDATED.json`. Each new case saves source and checkpoint provenance. Use fresh destinations: overwriting case directories does not safely resume.

All policies use simulated perception. The spatial alternatives intentionally cover the entire field with a **12×32×64 semantic raster**, while entity alternatives receive retained perception tracks. No architecture receives hidden FUEL or opponent truth. Pickup scenarios mask scoring, interception, and ferry actions 4/5/6 to prevent reacquisition loops. Earlier unrestricted-pickup results are under `evaluations/architecture-study-20261020/outdated-unrestricted-pickup`, with `OUTDATED.json`, and are excluded from comparisons.

## Launch the complete two-GPU comparison

The runner launches each architecture/scenario/seed/rate in a fresh process, checks the frozen runtime hash before and after every case, records per-case logs, and refuses a nonempty matrix. The default invocation is a read-only plan. Full execution covers 105 main cases plus 14 separate learning-rate sensitivity cases.

```bash
python3 scripts/run_architecture_study_v5.py --run \
  --expected-source bfa926060795a3c97422ab15f1cee94a15cd31ca9ce84c79cee1e1cd0a5be32f
```

Run GPU qualification first with the frozen manifest (repeat with `--scenario defense_3v3`):

```bash
.venv/bin/python evaluations/architecture-study-20261020/privacy-audit/qualify_v5.py \
  --scenario offense_vs_deterministic_defense --device cuda:0 \
  --seed 100019 --ticks 400 \
  --expected-source evaluations/architecture-study-20261020/study-source-v5.json
```

Final dashboard replays require restarting both dashboard services and verifying served asset bytes; a passing unit suite alone does not establish simulated behavior. The qualification retains raw seeded repeats and checks actual acquisitions, scoring, motion and simulated opponent tracks; it does not measure a trained architecture's performance.

## Pickup acquisition screen

The default architecture list runs all seven networks. These settings give 640 learned-agent transitions per seed and scenario: four updates × 20 rollout decisions × eight worlds. Evaluate initialized, update-two, and final policies on four held-out worlds.

```bash
.venv/bin/python -m frc_defense.architecture_benchmark \
  --scenarios pickup_simple pickup_scattered \
  --seeds 7 19 43 --learning-rates 3e-4 \
  --envs 8 --eval-envs 4 --horizon 200 \
  --rollout-steps 20 --eval-ticks 200 \
  --updates 4 --eval-every 2 --action-repeat 10 \
  --device cuda:0 --disable-miopen \
  --output evaluations/architecture-study-20261020/matrix
```

Two otherwise identical AMD WX 9100 cards can split disjoint architecture lists using `--architectures` and `--device cuda:0` / `cuda:1`. Never assign the same architecture/scenario/seed/learning-rate tuple to simultaneous jobs. Sample-learning plots combine matching protocols across slots; timing is descriptive because overlapping jobs affect performance.

## Two defense scenarios

These are independent training runs on deterministic opponents, not a transfer curriculum. Forty rollout decisions at ten physics ticks each match a 400-tick horizon. With three learned agents, `defense_3v3` counts three times as many agent transitions as the one-defender task at the same world budget.

```bash
.venv/bin/python -m frc_defense.architecture_benchmark \
  --scenarios deterministic_defense defense_3v3 \
  --seeds 7 19 43 --learning-rates 3e-4 \
  --envs 8 --eval-envs 4 --horizon 400 \
  --rollout-steps 40 --eval-ticks 400 \
  --updates 4 --eval-every 2 --action-repeat 10 \
  --device cuda:0 --disable-miopen \
  --output evaluations/architecture-study-20261020/matrix
```

Compare initialized and final behavior, terminal attacker score, and red score integral. The integral measures scoring delay within the fixed horizon, even when terminal scores match. First-event times are sampled at action-repeat resolution; null means the event was not reached. A hold tactic is not a physically absent defender: establish a separate undefended-attacker baseline before attributing score prevention. Zero task events during training limit what a short defense screen can establish.

## Experimental offense against deterministic defense

`offense_vs_deterministic_defense` uses a research factory with blue learned offense and a red deterministic defender. Generic neural strategic execution lacks deterministic-only intake/scoring refinements. Qualify this path before adding it to the main matrix:

```bash
.venv/bin/python -m frc_defense.architecture_benchmark \
  --architectures baseline_mlp \
  --scenarios offense_vs_deterministic_defense \
  --seeds 7 --learning-rates 3e-4 \
  --envs 2 --eval-envs 2 --horizon 600 \
  --rollout-steps 60 --eval-ticks 600 \
  --updates 0 --eval-every 2 --action-repeat 10 \
  --device cuda:0 --disable-miopen \
  --output evaluations/architecture-study-20261020/qualification/offense
```

The 600-tick (12-second) offense horizon was declared before the V5 rerun and retained uniformly. The current V5 scripted neural-path qualification at seed 100019 scores blue points `[0, 3, 5, 6]` over four worlds within 400 ticks. An initialized greedy network need not score; the zero-update CLI above checks construction and execution, while `privacy-audit/qualify_v5.py` checks actual scripted acquisition/scoring. The matrix command is:

```bash
.venv/bin/python -m frc_defense.architecture_benchmark \
  --scenarios offense_vs_deterministic_defense \
  --seeds 7 19 43 --learning-rates 3e-4 \
  --envs 8 --eval-envs 4 --horizon 600 \
  --rollout-steps 60 --eval-ticks 600 \
  --updates 4 --eval-every 2 --action-repeat 10 \
  --device cuda:0 --disable-miopen \
  --output evaluations/architecture-study-20261020/matrix
```

Use distinct output roots for alternative horizons, because the case directory name does not encode horizon. Do not infer an architecture failure from a controller limitation.

## Backend and learning-rate controls

HIP runs use `--disable-miopen` consistently. The residual CNN triggered a MIOpen `wrw_reduction_hip` weight-gradient fault on this hardware. The alternate-backend GPU qualification at `evaluations/architecture-study-20261020/qualification/neural-backend-disabled.json` covers the three field encoders at batches 32 and 96. All seven actual learning-matrix alternatives use the selected backend. This is a backend workaround, not an architectural result.

The declared extra sweep runs all seven networks on focused deterministic defense, seed 7 only, at 1e-4 and 1e-3, after the main 3e-4 cases. It remains a uniform sensitivity check after invalidating the earlier matrix, with no architecture-specific tuning. This adds 14 cases to the 105-case main matrix (119 expected overall):

```bash
.venv/bin/python -m frc_defense.architecture_benchmark \
  --scenarios deterministic_defense \
  --seeds 7 --learning-rates 1e-4 1e-3 \
  --envs 8 --eval-envs 4 --horizon 400 \
  --rollout-steps 40 --eval-ticks 400 \
  --updates 4 --eval-every 2 --action-repeat 10 \
  --device cuda:1 --disable-miopen \
  --output evaluations/architecture-study-20261020/matrix
```

Keep learning rates separate. This one-seed sweep is exploratory sensitivity analysis, not nested validation or evidence of an independently optimal learning rate. A future tuning study needs validation seeds before a final held-out test. The main three-seed short screen supports triage, not statistical superiority claims. Initialization/perception noise, rewards, actions, and low-level controllers remain material controls.

## References and frozen-policy map diagnostics

The completed instrumented absence references are under `evaluations/architecture-study-20261020/references-absent-blue-v5-gpu`. To reproduce into a fresh destination:

```bash
.venv/bin/python scripts/evaluate_architecture_undefended.py \
  --device cuda:0 --scenarios deterministic_defense \
  --seeds 100019 --envs 4 --ticks 400 --repeat-check \
  --expected-source evaluations/architecture-study-20261020/study-source-v5.json \
  --output evaluations/new-absence-reference
```

Repeat in a separate fresh process for each of the two scenarios and seeds 100007/100019/100043, as in the accepted study. A trajectory tolerance failure is retained with both records; do not rerun a failed case merely to obtain a passing result, or silently omit it. Removing observers changes sensor batching and the instrumented eager-physics path differs from the matrix; these are descriptive references.

The CNN diagnostic defaults to inspection only. This executes one paired normal/masked-object-map evaluation of an existing final checkpoint, into a fresh output root:

```bash
.venv/bin/python scripts/evaluate_cnn_field_sensitivity.py --run \
  --scenario pickup_scattered --seed 19 --device cuda:0 \
  --expected-manifest evaluations/architecture-study-20261020/study-source-v5.json \
  --output evaluations/new-cnn-diagnostic
```

The accepted diagnostic covers every task and training seed (15 pairs), with exact masked categorical decision traces. The intervention removes only raster channels 2/3/4/5/8/11 and retains the scalar branch and static/own/team/HUB map. It does not retrain a map-free network. Review both outcome parity and full frame parity: the completed study has one material fresh-normal trajectory mismatch despite exact outcome metrics. Full trace parity is available for the paired diagnostic, but not for the original main evaluation.

`scripts/export_architecture_replay.py` writes native dashboard bundles with actual executed actions, checkpoint/runtime hashes, sensors and HUB phases. The completed four exports and seeded record reviews are in `replays-v5/exports.json` and `replays-v5/review-summary.json`. The two 160-second cases use training seed 19, unseen evaluation seed 200019 and the fixed update 4 baseline/CNN checkpoints. They are illustrative transfer checks from eight-second training, not full-match training or a seven-model ranking. Restart and verify dashboard services before generating a new replay.

## Summarize and review

```bash
python3 scripts/summarize_architecture_study.py \
  --root evaluations/architecture-study-20261020/matrix \
  --output docs/NN_ARCHITECTURE_RESULTS.md
```

Use the system `python3`: its matplotlib installation renders standalone SVG and PNG research curves with the noninteractive Agg backend. The virtual environment lacks matplotlib and falls back to dependency-free SVG. The script includes completed cases only, separates learning rates and protocols, and preserves raw seed metrics. Compact charts include only scenarios present in that protocol: pickup plots show acquisitions, defense plots show net score and red score integral, and offense plots show acquisitions and net score. Tables retain every metric. It excludes named qualification/integration/outdated trial trees and cases under an `OUTDATED.json` marker. Device slots remain in per-case rows but do not split matching sample-learning plots. Full settings, evaluation records, references, and source provenance are linked in the report. Treat intermediate reports as snapshots; the study is not final until all 105 main cases and the declared learning-rate sweep have completed and been reviewed.

Before generating or reviewing dashboard replays, restart the serving services on the frozen checkout, verify active services and successful endpoints, and verify current frontend assets. Generate fresh records; label old-code replays outdated. Report scenario, seed, record identifier, tests, simulator outcome, and dashboard freshness separately. This summary script reads recorded metrics and does not perform those validation steps.

The [research review](NN_ARCHITECTURE_RESEARCH.md) explains architecture hypotheses, limitations, fair ablations, and future memory/graph/KAN experiments. No architecture conclusion is available from infrastructure qualification alone.
