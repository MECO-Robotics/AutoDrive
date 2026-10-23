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

The held-out layouts shift real bump and trench collision/observation geometry.
Held-out defender families run distinct orbiting-interceptor and split-chaser
commands; start lanes and sensor noise also vary in the simulator. These case
changes have only received static checks in this implementation. They have
not been validated in an end-to-end simulator evaluation.

Gate values are frozen in the protocol but remain provisional until a
scripted-reference baseline variance review is completed. The dashboard shows
that calibration state alongside any passing streak; a reported promotion is
conditional on those provisional thresholds.

Final-test execution is separate and requires the already promoted scenario,
the same protocol, and the exact checkpoint bundle recorded at promotion. It
uses only the selected candidate plus D0 and never changes promotion state.
Do not regenerate or inspect final-test reports while selecting among policy
checkpoints.
