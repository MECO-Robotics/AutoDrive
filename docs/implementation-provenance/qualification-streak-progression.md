# Qualification streak progression

## Planner

Scheduled validation evaluates updated policies, so requiring identical checkpoint IDs across rounds prevented the three-pass streak from advancing. The qualification protocol remains fixed across a streak; target and prerequisite retention evidence must bind the same checkpoint bundle within each round. A failed round or protocol change resets the streak. The third passing round supplies the checkpoint used for final testing.

## Implementer

Commit `e9faa56978fe7a74a00ecd2813e094fb8e81ba66` removes checkpoint identity from the cross-round streak comparison while recording the latest passing bundle and promoting the third round's checkpoint. Duplicate evaluation IDs now leave state unchanged. The qualification guide clarifies cross-round checkpoint updates and within-round bundle matching. Tests cover three passes with changing checkpoint IDs, protocol reset, duplicate no-op, and third-round checkpoint selection.

## Reviewer

Independent review approved exact commit `e9faa56978fe7a74a00ecd2813e094fb8e81ba66`. The within-round target/retention bundle check remains enforced, final-test binding uses the promoted checkpoint, and the tests cover the changed streak behavior.

## Evidence limits

The focused contract suites passed (74 tests). This change concerns qualification bookkeeping and does not claim simulated gameplay validation or a dashboard replay.
