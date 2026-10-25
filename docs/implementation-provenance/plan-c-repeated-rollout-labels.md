# Plan C matched repeated-rollout labels

This tranche implements the Plan §5 requirement to account for simulator
variability when generating candidate gross-return preferences for C1/C2. It
is based on source commit `9bf7d8d55e75821b67eb1c5d1c01871f666ea85d` and is an
offline implementation candidate. It does not collect labels, fit checkpoints,
run the simulator, or establish policy quality.

Implementation commit: `3186636a559a7729387a4b7989ca6caee0569b4d`.

For each archived Plan C candidate and eligible prefix, the collector now runs
exactly two branch repeats. Every candidate at the same prefix shares the same
two repeat RNG seeds. Each branch first reconstructs the archived prefix from
the original train seed and then enters a repeat-specific RNG scope for the
candidate branch and teacher continuation. At that branch point, the collector
also reseeds every simulator-local Torch generator by its stable environment
attribute path and records a digest of those generator seeds; candidates must
share the same per-repeat generator manifest. The repeat seed is derived from a
canonical JSON identity containing the repeat schema, qualification protocol
ID, scenario, category, base seed, decision index, and repeat index. The
manifest binds the repeat count, derivation, common-random-number rule,
continuation policy, and preference policy under a repeat-protocol digest.

The dataset retains every repeat's gross macro return, gross horizon return,
discounted horizon return, branch terminal flag, continuation count, repeat
index, and derived seed. Candidate means and sample variances are recorded.
For each candidate pair, the artifact records the matched repeat differences,
mean difference, sample variance, standard error, and stability. A pair is
trainable only when both repeat differences have the same strict sign;
otherwise the fitter abstains. Missing or reordered repeats, incorrect seed
binding, altered aggregates, altered variance, or forged pair labels fail
closed. C1/C2 fitting rejects legacy single-run Plan C labels. A1/A2/A3's
existing v2 fitting path is unchanged.

`R=2` is a bounded replication policy chosen for this tranche. It supports a
conservative sign-stability rule but does not provide precise variance
calibration, confidence intervals, or qualification evidence. Reviewer
follow-up adds collector-level matched-seed flow coverage using mocked
environment, graph, reward, and branch interfaces; explicit C1/C2 legacy
manifest rejection checks; and repeated-artifact metadata tamper rejection
checks. These tests do not run actual game behavior. GPU collection and a
representative end-to-end simulator evaluation remain unrun; no replay or
dashboard-currentness claim is made.
