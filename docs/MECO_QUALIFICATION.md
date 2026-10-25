# MECO qualification protocol

## Calibrate before validating

New promotion runs use a target-scoped calibrated v5 GPU-bootstrap protocol. It freezes one
matched deterministic reference before held-out evaluation: `D1/current`,
`D2/cv`, or `D2/multi_model_v1`. D0 remains a separately reported safety floor.
Historical v1 protocols are nonpromotable; legacy calibrated v2 and v3
protocols and their reports remain readable with their original comparison
semantics. Do not run promotion validation until the target's complete
prerequisite closure has disjoint D0 calibration reports, a verified bundle,
repeated matched-reference variance reports, an explicit reviewed margin
decision, and an immutable v5 protocol using implementation ID
`meco-v4-cluster-episode-device-percentile-v1`. Versions v1-v4 are historical
or diagnostic only for the five-track Stage 1 gate.

Calibration, model evaluation, and qualification currently require an
explicitly indexed available ROCm/HIP GPU, exposed by PyTorch with a device name
such as `cuda:0`. The native fuel extension rejects an NVIDIA CUDA runtime;
support for CUDA without HIP is not implemented. `cpu`, the unindexed `cuda`
alias, and unavailable device indices fail before MECO compute starts. The
simulator/model/evaluator compute runs on the selected GPU. Python orchestration,
artifact serialization, and checkpoint `map_location`/RNG handling may use host
CPU; this GPU-only contract does not claim zero host CPU activity.

The current user requirement is stricter than that legacy contract: algorithmic
training, candidate generation/scoring, route/risk calculation, and statistical
evaluation must stay on the GPU. The isolated candidate has moved only D1/D2
scorer arithmetic onto the episode device; other host-side decision and
analysis work remains. Do not treat device-selection checks as proof of
end-to-end GPU residency.

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

For long calibrations, use deterministic shards. The planner freezes the full
case table, settings, source identity, runtime, and exact `(repetition, case,
seed)` assignment in each immutable manifest. Each shard checkpoints complete
case/repetition groups to a digest-checked `.partial` file and resumes only
when that file still matches its manifest. A completed shard is immutable.
The shard CLI must prepare the native HIP extension and capture provenance in
the **same Python process**. A plain `python -m frc_defense.meco_qualification`
call can record an `unprepared` extension identity in the manifest, while
evaluation records the loaded binary identity; the shard will then reject its
own manifest. Use this helper for both manifest creation and shard execution:

```bash
run_prepared_meco_cli() {
  local gpu_device="$1"
  shift
  python -c 'import sys; from frc_defense.fuel_physics_hip import prepare_native_fuel_extension; prepare_native_fuel_extension(sys.argv[1]); from frc_defense.meco_qualification import main; main(sys.argv[2:])' \
    "$gpu_device" "$@"
}

# These paths identify the active 20261024 run in this checkout. For a new
# source identity or master seed, use a new source tag and campaign root.
source_tag=source-31e5
scenario_slug="${scenario//_/-}"
campaign_root="qualifications/meco/shards/${scenario_slug}-20261024/${source_tag}"
# Generate a full, device-bound manifest set for each intended ordinal.
devices=(cuda:0 cuda:1)
for shard_device in "${devices[@]}"; do
  run_prepared_meco_cli "$shard_device" plan-d0-calibration-shards \
    --scenario "$scenario" --episodes 100 --repetitions 3 \
    --master-seed 20261024 --shard-count 8 --device "$shard_device" \
    --manifest-dir "$campaign_root/${shard_device//:/-}"
done
```

Generate one shard artifact per assigned index from its matching device
manifest, then merge the eight artifacts into the ordinary immutable v2
calibration report before bundling:

```bash
run_prepared_meco_cli() {
  local gpu_device="$1"
  shift
  python -c 'import sys; from frc_defense.fuel_physics_hip import prepare_native_fuel_extension; prepare_native_fuel_extension(sys.argv[1]); from frc_defense.meco_qualification import main; main(sys.argv[2:])' \
    "$gpu_device" "$@"
}

# Assign each index exactly once, using the matching ordinal's manifest.
# This balanced example assigns four of eight shards to each WX9100.
declare -A shard_device=(
  [0]=cuda:1 [1]=cuda:1 [2]=cuda:0 [3]=cuda:1
  [4]=cuda:0 [5]=cuda:1 [6]=cuda:0 [7]=cuda:0
)
for shard_index in {0..7}; do
  shard_device_for_index="${shard_device[$shard_index]}"
  device_dir="$campaign_root/${shard_device_for_index//:/-}"
  output_dir="evaluations/meco-five/d0-calibration-20261024/${source_tag}/${shard_device_for_index//:/-}"
  shard_id=$(printf 'd0-calibration-%s-shard-%04d-of-0008' "$scenario" "$shard_index")
  shard_file=$(printf 'shard-%04d.artifact.json' "$shard_index")
  manifest="$device_dir/${shard_id}.json"
  run_prepared_meco_cli "$shard_device_for_index" calibrate-d0-shard \
    --manifest "$manifest" --device "$shard_device_for_index" \
    --output "$output_dir/$shard_file"
done
shards=()
for shard_index in {0..7}; do
  shard_device_for_index="${shard_device[$shard_index]}"
  output_dir="evaluations/meco-five/d0-calibration-20261024/${source_tag}/${shard_device_for_index//:/-}"
  shard_file=$(printf 'shard-%04d.artifact.json' "$shard_index")
  shards+=(--shard "$output_dir/$shard_file")
done
run_prepared_meco_cli "$DEVICE" merge-d0-calibration-shards \
  "${shards[@]}" --output "qualifications/meco/d0-calibration-${scenario}.json"
```

The merge requires every shard index exactly once, checks that each assigned
tuple appears once and that no unexpected tuple appears, validates source,
controller, safety, and normalized runtime provenance, and recomputes the v2
summary from raw merged records. Shards may use different indexed GPU ordinals
only when backend, GPU model, and the remaining runtime identity match. The merged
v2 report records normalized runtime identity and each row's exact shard
device. Shard manifests and shard outputs are not accepted as calibration
reports by protocol creation or bundle intake.

Each manifest is bound to the exact indexed device named in
`runtime_device`; invoke its shard on that same ordinal. To split work across
devices, generate the shard set on each intended ordinal and use the
manifest for each shard index that matches the executing device. Keep the
checkout, Python environment, HIP extension path/binary, and toolchain fixed
from manifest creation through shard completion; changing them invalidates
the captured source or runtime identity.

The D0 bundle is rejected if any prerequisite is missing, extra, duplicated,
or has an overlapping calibration seed pool.

Collect matched-reference variance evidence for every scenario in the target's
prerequisite closure, using the same scenario seed table as its D0 report.
Each repeat evaluates the selected D1/D2 reference and D0. D0 remains a
separate safety floor; these paired rows do not replace its immutable
calibration report. The runner rejects non-indexed devices, missing safety
metrics, source/runtime drift, policy/mode/contract drift, and calibration
seeds that overlap train, validation, or test.
The calibration evaluator uses a frozen candidate configuration of K=16,
the training default route-cost weight, and 12 physics ticks per action. Its
explicit `calibration_control_policy` opt-in accepts only an empty checkpoint
set, empty learned-policy selection, D0 plus exactly the selected D1/D2
control, a fixed stratum, and no inference fault or replay output. Ordinary
evaluation and D0-only calibration retain their existing validation rules.

```bash
TARGET=pickup_simple
REFERENCE=D1/current
DEVICE=cuda:0
mapfile -t scenarios < <(python -c 'import sys; from frc_defense.meco_qualification import _scenario_closure; print("\n".join(_scenario_closure(sys.argv[1])))' "$TARGET")
reference_reports=()
for scenario in "${scenarios[@]}"; do
  report="qualifications/meco/reference-variance-${scenario}.json"
  python -m frc_defense.meco_qualification calibrate-matched-reference \
    --scenario "$scenario" --reference-policy "$REFERENCE" --episodes 100 \
    --repetitions 3 --master-seed 20261024 --device "$DEVICE" --output "$report"
  reference_reports+=(--report "$report")
done
python -m frc_defense.meco_qualification bundle-matched-reference-calibrations \
  "${reference_reports[@]}" \
  --output "qualifications/meco/reference-variance-${TARGET}-bundle.json"
```

Review the variance evidence and provide final margins explicitly. Protocol
creation does not calculate operational margins from observed variance. The
review file binds the exact report IDs and hashes, records reviewer identity
and timestamp, and declares the final useful-gain margins plus safety margins:
The report ID and hash maps contain one entry for every scenario in the
prerequisite closure; the example shows one entry for brevity.

```json
{
  "schema_version": 1,
  "reviewed_by": "reviewer name or role",
  "reviewed_utc": "2026-10-24T12:00:00Z",
  "decision": "accepted",
  "reference_variant": "D1/current",
  "reference_calibration_ids": {"pickup_simple": "<64 hex chars>"},
  "reference_report_sha256": {"pickup_simple": "<64 hex chars>"},
  "minimum_useful_gain": {"single": 0.05, "cluster": 0.03},
  "safety_noninferiority_margins": {
    "safety_robot_contact_rate": 0.01,
    "safety_field_contact_rate": 0.01,
    "safety_wall_contact_rate": 0.01
  }
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
finite nonnegative noninferiority margins in [0, 1]. The margin review must
repeat these values exactly.

Choose the matched reference before any held-out evaluation. The old `D1` and
`D2` CLI spellings remain accepted as aliases for `D1/current` and `D2/cv`.
The selected variant, exact predictor contract, policy ID, and D0 safety-floor
role are included in the protocol identity.

```bash
TARGET=pickup_simple
REFERENCE=D1/current
python -m frc_defense.meco_qualification create-protocol \
  --scenario "$TARGET" --episodes 100 --master-seed 20261024 \
  --reference-policy "$REFERENCE" \
  --calibration-bundle-json "qualifications/meco/d0-calibration-${TARGET}-bundle.json" \
  --matched-reference-calibration-bundle-json "qualifications/meco/reference-variance-${TARGET}-bundle.json" \
  --reference-margin-review-json "qualifications/meco/reference-margin-review-${TARGET}.json" \
  --phase0-identity-binding-json qualifications/meco/stage1-phase0-identity-binding.json \
  --safety-margins-json qualifications/meco/safety-margins.json \
  --output "qualifications/meco/protocol-${TARGET}-v5.json"
```

Replace `pickup_simple` with the selected promotion target. The protocol binds
the target and all transitive prerequisites, exact train/validation/test seed
lists, D0 and matched-reference report IDs and hashes, source/runtime identity,
the frozen matched reference and prediction contract, safety schema, and
explicit reviewer-approved useful-gain and safety margins. D0 remains the
separately bound safety floor. Existing protocol files are never rewritten.

## Stage 1 metadata readiness preflight

Before launching the matched Stage 1 campaign, run `preflight-stage1` with
the exact v5 protocol, a digest-pinned Phase 0 evidence index, the
qualification campaign manifest, and one manifest for each of A1, A2, A3, C1,
and C2. Supply every active/protected campaign and checkpoint root. The
campaign manifest also declares its campaign/checkpoint roots; these are added
to the protected set before any Phase 0 evidence path is opened.

```bash
python -m frc_defense.meco_qualification preflight-stage1 \
  --protocol qualifications/meco/protocol-pickup_simple-v5.json \
  --phase0-index qualifications/meco/stage1-phase0-evidence-index.json \
  --campaign-manifest qualifications/meco/stage1-campaign-manifest.json \
  --checkpoint-manifest qualifications/meco/checkpoints/A1-manifest.json \
  --checkpoint-manifest qualifications/meco/checkpoints/A2-manifest.json \
  --checkpoint-manifest qualifications/meco/checkpoints/A3-manifest.json \
  --checkpoint-manifest qualifications/meco/checkpoints/C1-manifest.json \
  --checkpoint-manifest qualifications/meco/checkpoints/C2-manifest.json \
  --protected-campaign-root /data/meco/active-campaign \
  --protected-checkpoint-root /data/meco/checkpoints \
  --output qualifications/meco/stage1-readiness.json
```

The Phase 0 index uses schema `meco-stage1-phase0-evidence-index-v1`; it binds
the evaluator `source_identity_sha256` separately from the freeze's Git
`commit`, `tree`, `source_plan_path`, and `source_plan_sha256`. It contains
relative path and exact-byte SHA-256 references for the capture receipt,
effective snapshot, effective freeze, tuple registry, partition materialization,
score-estimand review, D0 calibration bundle, matched-reference calibration
bundle, and margin review. Its canonical `index_sha256` covers those references
and the explicit runtime, dependency-lock, normalized-runtime, and eight
configuration identities. For a new calibrated `pickup_simple` Stage 1
protocol, `phase0_identity_binding` is required in the v5 protocol body and is
covered by `protocol_id`; the campaign must carry the identical binding. The
preflight checks the serialized receipt→snapshot→freeze→tuple registry/partition
chain and each embedded digest/link. It does not re-capture source/runtime state,
reproduce the factory, inspect the current checkout, or execute the full Phase 0
materializer. Those freshness checks are explicitly reported as
`current_source_factory_reproduction: not_performed`.
The baseline-freeze contract currently has an empty trusted allocation-artifact
digest allowlist. An absent authorization or an unpinned authorization digest
therefore blocks readiness (`phase0_allocation_authorization_not_validated` or
`phase0_allocation_authorization_untrusted`) until the separately reviewed
approval digest is pinned in the canonical contract.
The campaign manifest uses schema
`meco-five-track-stage1-campaign-manifest-v1`; each track manifest uses
`meco-stage1-checkpoint-manifest-v1`. These metadata contracts bind the
scenario/version, source/config/protocol, disjoint validation/final-test seed
sets, selected single comparator, D0 safety floor, and common five-track
campaign ancestry. The campaign manifest fields are `qualification_eligible`,
`evaluation_split`, `scenario_id`, `scenario_version`, `protocol_id`,
`source_identity_sha256`, `config_sha256`, `promotion_comparator`,
`safety_floor_policy`, `validation_seeds`, `final_test_seeds`,
`matched_campaign_ancestry`, `campaign_roots`, `checkpoint_roots`, and
`phase0_identity_binding`, in addition to its schema tag. Checkpoint manifests
carry `policy_id`, the declared `checkpoint_sha256`, source/config/protocol/scenario
bindings, both seed-partition digests, and `matched_campaign_ancestry`, in
addition to their schema tag. Campaign-declared roots must be absolute paths
that resolve to existing directories. Input and referenced evidence paths are canonicalized through
symlinks and rejected if they fall inside any protected root; the report path
is subject to the same check. The output is written only to the explicit
destination, and an existing non-identical report is never replaced.

The report is deterministic for the same inputs and uses stable blocker codes.
Expected input-readiness failures (for example, a missing campaign manifest,
missing or malformed campaign root declarations, protected input paths, or
protected referenced evidence) produce a `blocked` report when the output path
has been validated outside all known protected roots. Paths deliberately not
opened have a null evidence digest. Unresolvable protected roots and unsafe
output paths fail closed without writing a report.
`ready: true` means only `metadata_consistent_only`: it does not open, stat,
hash, or deserialize checkpoint bytes, execute models, read held-out outcomes,
advance curriculum state, qualify a track, or authorize GPU work. Checkpoint
SHA-256 values are declarations in the manifests and are explicitly unverified
against file bytes. Diagnostic campaign manifests, including the active r4
campaign, remain blocked. The preflight is read-only with respect to campaign
and checkpoint data and does not imply permission to interrupt or compete with
the currently active training campaign.

The output schema is `meco-stage1-readiness-preflight-v1`; it contains
`status`, `ready`, `scope`, `checkpoint_digest_semantics`, explicit false
side-effect/outcome flags, `current_source_factory_reproduction`, five separate
`tracks` rows, per-check evidence paths and digests, and stable `blocker_codes`.
Illustrative output fragments (not evidence):

```json
{"status":"blocked","ready":false,"scope":"metadata_consistent_only",
 "current_source_factory_reproduction":"not_performed",
 "blocker_codes":["phase0_artifact_set_incomplete"]}
```

```json
{"status":"ready","ready":true,"scope":"metadata_consistent_only",
 "current_source_factory_reproduction":"not_performed",
 "checkpoint_digest_semantics":"declared_metadata_only_not_verified_against_checkpoint_bytes",
 "blocker_codes":[]}
```

The second fragment represents metadata consistency only; it is not a report
that a real campaign or any model passed.

## Run validation under the frozen v5 protocol

Validation requires at least 100 episodes per category and each frozen
layout/family stratum. Within each round, use the exact same protocol and
checkpoint bundle for the target and every transitive prerequisite. Run
prerequisite scenarios as retention evidence, then run the target to advance
its promotion streak. Each v5 round pairs the selected learned policy with the
preselected D1/D2 reference on identical seeds, and separately checks safety
against D0. Useful-throughput and safety confidence intervals use the frozen
reference; no reference is selected after observing held-out outcomes. Keep a
separate state file for each candidate policy.

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
  --protocol qualifications/meco/protocol-pickup_simple-v5.json \
  --state qualifications/meco/A1-state.json \
  --output qualifications/meco/A1-pickup-simple-round-1.json \
  --device cuda:0
```

Promotion requires three consecutive passing target rounds under the same
frozen protocol. Checkpoint bundles may advance between rounds as the candidate
is updated; each round's target and every prerequisite retention report must
still use the exact same bundle. The third passing round's bundle is recorded
as the qualified checkpoint and is the one bound to final-test evaluation. A
failed round or protocol change resets the streak. Promotion state is staged
until the immutable round report is written and state is atomically saved. An
aborted round restores prior qualification and evaluation history, resets only
that scenario's streak, records the failure, and refreshes dashboard status.
Per-episode coverage validation checks scenario identity.

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
policy manifest, source identity, cases, safety provenance, and raw rows. New
evidence schema v2 also binds final-test summary fields; older reports without
an explicit evidence schema retain the legacy digest field set, and calibrated
v3 through v5 reports require schema v2. V4 evidence additionally binds
matched-reference variance reports and the explicit margin review; v5 uses
the frozen GPU cluster-bootstrap implementation.
Unknown `safety_*` row fields fail closed. These unkeyed SHA-256 checks detect
edits while the recorded digest is retained; they do not authenticate the
producer. No signing key or trusted attestor is configured. Runtime calibration
and representative simulator replay are still required; schema readiness does
not establish runtime parity or promotion eligibility.

Final-test execution is separate and requires an already promoted scenario,
the same protocol, and the exact checkpoint bundle recorded at promotion. A
v3-v5 test includes the selected candidate, its preselected D1/D2 reference, and D0
as a separate safety floor; legacy v2 keeps its D0 comparison. Final-test
results are digest-bound and never change promotion state. Do
not regenerate or inspect final-test reports while selecting among policy
checkpoints.

Diagnostics remain available through `python -m frc_defense.meco_track_training
evaluate`, but they never update promotion state and require an explicit
`--device cuda:N`.

The five-track training manifest can bind v1 through v4 protocol bytes for
training and validation planning. Its calibration projection does not validate
the required calibration artifacts, so it always reports qualification
eligibility as false. Use the qualification protocol and report validators for
promotion; the training manifest is not a qualification admission path.

New five-track Stage 1 campaign qualification claims require a v5 GPU-bootstrap
protocol with implementation ID `meco-v4-cluster-episode-device-percentile-v1`, the exact
`matched_reference_variance_and_reviewed_margin_frozen` status and must pass
the full protocol validator. Legacy v2/v3 protocols remain readable and may
support explicitly diagnostic validation plans, but those plans are never
qualification-eligible.
