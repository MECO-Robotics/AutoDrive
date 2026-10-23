# MECO qualification protocol

Diagnostics remain available through `python -m frc_defense.meco_track_training evaluate`; they never update promotion state. Frozen qualification uses a separate command:

```bash
python -m frc_defense.meco_qualification create-protocol \
  --output qualifications/meco/protocol.json --master-seed 20261024

python -m frc_defense.meco_qualification run-validation \
  --scenario pickup_simple --candidate-policy A1 \
  --checkpoint checkpoints/meco-five/A1/latest.pt \
  --protocol qualifications/meco/protocol.json \
  --state qualifications/meco/A1-state.json \
  --output qualifications/meco/A1-pickup-simple-round-1.json --device cpu
```

Create the protocol once and retain it unchanged. It freezes exact train,
validation, and final-test seed lists. The continuous `MecoTaskScheduler` uses
the protocol's 100-seed train pool by scenario and records the pool hash and
selected seed on each task assignment. Set the protocol master seed equal to
the supervisor `--seed`. Training manifests record the assignment; the exact
pool itself is in the protocol artifact.

Validation uses at least 100 episodes per category and per layout/family
stratum, with the selected track paired against deterministic D0 on identical
seeds. It records immutable per-episode outcomes, checkpoint/source hashes,
paired stratified bootstrap intervals, and exact case coverage. Each round
evaluates all transitive prerequisites on the same selected checkpoint and
protocol. Promotion state resets a streak on any failure, protocol change, or
checkpoint-bundle change, and promotes after three consecutive passing rounds.
Keep a separate state file for each candidate policy.

Promotion state is staged until the immutable round report is written and the
state file is atomically saved. An aborted round restores prior qualification
and evaluation history, resets only that scenario's streak, records the failure,
and refreshes dashboard status. Per-episode coverage validation also requires
each record's scenario identity to match its report.

The held-out layouts shift real bump and trench collision/observation geometry.
Held-out defender families run distinct orbiting-interceptor and split-chaser
commands; start lanes and sensor noise also vary in the simulator. These case
changes have only received static checks in this implementation. They have
not been validated in an end-to-end simulator evaluation.

Gate values in the legacy v1 protocol are provisional. Such a protocol remains
readable for historical reports, but it cannot advance promotion. Calibrate D0
on a dedicated seed partition, then freeze a target-scoped v2 protocol with
reports for the target and every transitive prerequisite, plus explicit useful
gain and safety noninferiority margins. For a target with prerequisites, run
`calibrate-d0` once per scenario in its prerequisite closure and bundle those
immutable reports:

```bash
python -m frc_defense.meco_qualification calibrate-d0 \
  --scenario pickup_simple --episodes 100 --repetitions 3 \
  --master-seed 20261024 --device cuda:0 \
  --output qualifications/meco/d0-calibration-pickup-simple.json

python -m frc_defense.meco_qualification bundle-calibrations \
  --report qualifications/meco/d0-calibration-pickup-simple.json \
  --output qualifications/meco/d0-calibration-pickup-simple-bundle.json

python -m frc_defense.meco_qualification create-protocol \
  --scenario pickup_simple --episodes 100 --master-seed 20261024 \
  --calibration-bundle-json qualifications/meco/d0-calibration-pickup-simple-bundle.json \
  --minimum-useful-gain-json qualifications/meco/useful-gain-margins.json \
  --safety-margins-json qualifications/meco/safety-margins.json \
  --output qualifications/meco/protocol-pickup-simple-v2.json
```

Calibration emits the immutable `meco-d0-calibration-v2` report format and
requires an explicit visible GPU device (`cuda:0`, `cuda:1`, etc.)
and records the selected device, backend, and runtime version. CPU calibration
fails closed. Calibration repeats the same seed list independently to separate repeat-run
variation from across-seed variation. Calibration seeds use split code 3 and
are checked against train, validation, and test seeds. Existing v1 protocol
files are not rewritten. The v2 digest binds the calibration report hashes,
source identity, per-scenario seed pools and gates, gain-margin derivation, and
safety tolerances. Set the operational useful-gain threshold explicitly for
each target category; baseline variance only adds a confidence allowance and
never defines the desired gain. For example:

```json
{
  "schema_version": 1,
  "operational_gain_threshold": {"single": 0.05, "cluster": 0.03},
  "uncertainty_multiplier": 1.96
}
```

The v2 margin is `operational_gain_threshold + uncertainty_multiplier ×
maximum_stratum_standard_error`, with standard error derived from calibration
between-seed and within-seed repeat variance. The declaration, derived values,
method, calibration ID, and report hashes are all bound into the protocol and
qualification evidence. Safety tolerances remain explicitly declared per
supported rate and are frozen into the same protocol. Calibration collection
does not currently expose episode sharding across devices; concurrent jobs
must use distinct report paths, and bundle intake rejects overlapping seed
pools.
It contains exact train/validation/test lists for the target and prerequisite
closure, and verifies calibration pools are disjoint from all three splits and
from each other. Only the target can advance promotion; prerequisite runs are
collected as retention evidence under the same protocol and checkpoint bundle.
Promotion requires each paired useful-throughput confidence interval to clear
its frozen gain margin. It also requires candidate-versus-D0 safety metrics
named by the frozen safety-margin file. Calibration intake verifies the exact
`meco-safety-rates-v1` schema, common source identity, record-level source
bindings, and all three finite per-active-tick rates. A calibrated v2 protocol
copies the verified schema and source identity into its hashed body and marks
`safety_gate_status` as `metrics_schema_verified`. Each validation category
subreport must match that frozen binding, and every candidate and D0 row must
contain exactly the three finite rates in [0, 1] with the report's source
identity. Missing or mismatched evidence fails closed; calibration alone does
not qualify a policy. Historical v1 rows remain readable under their separate
compatibility rules, but cannot advance promotion. Dashboard status reports
protocol version, calibration ID, and safety gate readiness.

Promotion consumption revalidates the complete v2 protocol schema, category
and safety metric name sets, schema/source binding, exact seed case table,
calibration/source IDs, and an `evidence_sha256` over the immutable report's
policy manifest, source identity, cases, safety provenance, and raw rows.
Unknown `safety_*` row fields are rejected. These are unkeyed SHA-256 integrity
checks: they detect edits when the recorded digest is retained, but they do not
authenticate the report producer or stop a forger from replacing both data and
digest. No signing key or trusted attestor is configured. Runtime calibration
and representative simulator replay remain necessary before treating observed
safety rates as empirical validation; schema readiness alone does not establish
runtime parity or promotion eligibility.

Final-test execution is separate and requires the already promoted scenario,
the same protocol, and the exact checkpoint bundle recorded at promotion. It
uses only the selected candidate plus D0 and never changes promotion state.
Do not regenerate or inspect final-test reports while selecting among policy
checkpoints.
