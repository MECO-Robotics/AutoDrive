# MECO qualification protocol

## Calibrate before validating

Promotion uses a target-scoped calibrated v2 protocol. Historical v1 protocol
files and reports remain readable for audit, but are **nonpromotable**. Do not
run promotion validation until the target's complete prerequisite closure has
disjoint D0 calibration reports, a verified bundle, explicit useful-gain and
safety margins, and an immutable v2 protocol.

Calibration, model evaluation, and qualification require an explicitly indexed
available CUDA or ROCm/HIP device such as `cuda:0`. `cpu`, the unindexed `cuda`
alias, and unavailable device indices fail before MECO compute starts. The
simulator/model/evaluator compute runs on the selected GPU. Python orchestration,
artifact serialization, and checkpoint `map_location`/RNG handling may use host
CPU; this GPU-only contract does not claim zero host CPU activity.

Choose the promotion target. This executable sequence prints its exact
prerequisite closure, collects one disjoint D0 report per scenario, and bundles
those reports. Set `TARGET` and `DEVICE` first:

```bash
TARGET=pickup_simple
DEVICE=cuda:0
mkdir -p qualifications/meco
mapfile -t scenarios < <(python -c 'import sys; from frc_defense.meco_qualification import _scenario_closure; print("\n".join(_scenario_closure(sys.argv[1])))' "$TARGET")
reports=()
for scenario in "${scenarios[@]}"; do
  report="qualifications/meco/d0-calibration-${scenario}.json"
  python -m frc_defense.meco_qualification calibrate-d0 \
    --scenario "$scenario" --episodes 100 --repetitions 3 \
    --master-seed 20261024 --device "$DEVICE" --output "$report"
  reports+=(--report "$report")
done
python -m frc_defense.meco_qualification bundle-calibrations \
  "${reports[@]}" --output "qualifications/meco/d0-calibration-${TARGET}-bundle.json"
```

The command requires at least 100 episodes and at least two repetitions.
Calibration seed pools are checked against train, validation, and test, and the
bundle rejects overlaps among reports. Keep the same source checkout and
runtime for the entire bundle.

The bundle is rejected if any prerequisite is missing, extra, duplicated, or
has an overlapping calibration seed pool.

Declare operational useful-gain thresholds per target category and the
uncertainty multiplier in `qualifications/meco/useful-gain-margins.json`:

```json
{
  "schema_version": 1,
  "operational_gain_threshold": {"single": 0.05, "cluster": 0.03},
  "uncertainty_multiplier": 1.96
}
```

Declare explicit noninferiority tolerances in
`qualifications/meco/safety-margins.json`:

```json
{
  "safety_robot_contact_rate": 0.01,
  "safety_field_contact_rate": 0.01,
  "safety_wall_contact_rate": 0.01
}
```

The safety file must use exactly the `meco-safety-rates-v1` metric names and
finite nonnegative noninferiority margins in [0, 1]. These values are
operational choices and must be reviewed for the target; calibration variance
does not choose the useful-gain threshold.

Create and retain the immutable target-scoped v2 protocol:

```bash
TARGET=pickup_simple
python -m frc_defense.meco_qualification create-protocol \
  --scenario "$TARGET" --episodes 100 --master-seed 20261024 \
  --calibration-bundle-json "qualifications/meco/d0-calibration-${TARGET}-bundle.json" \
  --minimum-useful-gain-json qualifications/meco/useful-gain-margins.json \
  --safety-margins-json qualifications/meco/safety-margins.json \
  --output "qualifications/meco/protocol-${TARGET}-v2.json"
```

Replace `pickup_simple` with the selected promotion target. The protocol binds
the target and all transitive prerequisites, exact train/validation/test seed
lists, calibration report IDs and hashes, source identity, safety schema,
operational thresholds, and derived gains. Its useful-gain margin is the
declared operational threshold plus the uncertainty multiplier times the
maximum stratum standard error estimated from between-seed and within-seed
repeat variance. Existing v1 protocol files are never rewritten.

## Run validation under the frozen v2 protocol

Validation requires at least 100 episodes per category and each frozen
layout/family stratum. Use the exact same protocol and checkpoint bundle for
the target and every transitive prerequisite. Run prerequisite scenarios as
retention evidence, then run the target to advance its promotion streak. Each
round pairs the selected policy with deterministic D0 on identical seeds and
checks useful-throughput confidence intervals and all frozen safety metrics.
Keep a separate state file for each candidate policy.

The continuous `MecoTaskScheduler` selects from the protocol's training seed
pool by scenario and records the pool hash and selected seed on each assignment.
Set the protocol master seed equal to the supervisor `--seed`; the exact pool is
frozen in the protocol artifact. D0 calibration repeats each scenario's same
seed list independently to estimate repeat-run variance separately from
across-seed variance.

```bash
python -m frc_defense.meco_qualification run-validation \
  --scenario pickup_simple --candidate-policy A1 \
  --checkpoint checkpoints/meco-five/A1/latest.pt \
  --protocol qualifications/meco/protocol-pickup-simple-v2.json \
  --state qualifications/meco/A1-state.json \
  --output qualifications/meco/A1-pickup-simple-round-1.json \
  --device cuda:0
```

Promotion requires three consecutive passing target rounds. A failed round,
protocol change, or checkpoint-bundle change resets the streak. Promotion
state is staged until the immutable round report is written and state is
atomically saved. An aborted round restores prior qualification and evaluation
history, resets only that scenario's streak, records the failure, and refreshes
dashboard status. Per-episode coverage validation checks scenario identity.

The held-out layouts shift bump and trench collision/observation geometry.
Held-out defender families run distinct orbiting-interceptor and split-chaser
commands; start lanes and sensor noise also vary in simulation. These case
changes still require representative end-to-end simulator evaluation and
inspection of a fresh replay before they can count as runtime evidence.

## Evidence and limits

Calibration emits immutable `meco-d0-calibration-v2` reports with device,
backend, runtime, source, seed pools, and variance evidence. Calibration itself
does not qualify a policy. Promotion revalidates the full protocol, category
and safety metric sets, schema/source binding, exact seed case table,
calibration/source IDs, and an `evidence_sha256` over the immutable report's
policy manifest, source identity, cases, safety provenance, and raw rows.
Unknown `safety_*` row fields fail closed. These unkeyed SHA-256 checks detect
edits while the recorded digest is retained; they do not authenticate the
producer. No signing key or trusted attestor is configured. Runtime calibration
and representative simulator replay are still required; schema readiness does
not establish runtime parity or promotion eligibility.

Final-test execution is separate and requires an already promoted scenario,
the same protocol, and the exact checkpoint bundle recorded at promotion. It
uses only the selected candidate plus D0 and never changes promotion state. Do
not regenerate or inspect final-test reports while selecting among policy
checkpoints.

Diagnostics remain available through `python -m frc_defense.meco_track_training
evaluate`, but they never update promotion state and require an explicit
`--device cuda:N`.
