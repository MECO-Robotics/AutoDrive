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
on a dedicated seed partition, then freeze a v2 protocol with explicit useful
gain and safety noninferiority margins:

```bash
python -m frc_defense.meco_qualification calibrate-d0 \
  --scenario pickup_simple --episodes 100 --repetitions 3 \
  --master-seed 20261024 --output qualifications/meco/d0-calibration-pickup-simple.json

python -m frc_defense.meco_qualification create-protocol \
  --scenario pickup_simple --episodes 100 --master-seed 20261024 \
  --calibration-report qualifications/meco/d0-calibration-pickup-simple.json \
  --minimum-useful-gain-json qualifications/meco/useful-gain-margins.json \
  --safety-margins-json qualifications/meco/safety-margins.json \
  --output qualifications/meco/protocol-pickup-simple-v2.json
```

Calibration repeats the same seed list independently to separate repeat-run
variation from across-seed variation. Calibration seeds use split code 3 and
are checked against train, validation, and test seeds. Existing v1 protocol
files are not rewritten. The v2 digest binds the calibration report, source
identity, gain margins, and safety margins. Promotion requires each paired
useful-throughput confidence interval to clear its frozen gain margin. It also
requires candidate-versus-D0 safety metrics named by the frozen safety-margin
file. Until the evaluator emits those safety metrics, v2 promotion fails closed
with an explicit missing-metric failure; calibration alone does not qualify a
policy. Dashboard status reports protocol version, calibration ID, and safety
gate readiness.

Final-test execution is separate and requires the already promoted scenario,
the same protocol, and the exact checkpoint bundle recorded at promotion. It
uses only the selected candidate plus D0 and never changes promotion state.
Do not regenerate or inspect final-test reports while selecting among policy
checkpoints.
