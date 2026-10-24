# Plan A offline U/B dataset audit: three-pass summary

## Planner pass

The planner found that Plan A collection and GPU fitting code already existed,
but no dataset had been collected, replay-validated, or fit. A read-only audit
was needed to bind stored legal-node U/B targets to their source replays and
protocol, recompute target/repeat evidence, and summarize sampling coverage
without claiming empirical results. The scope excludes simulator replay and
fitting; real collection and independent human labels remain separate plan
requirements.

## Implementer pass

Added `frc_defense/meco_plan_a_oracle_audit.py`, a fail-closed offline audit of
legal-node datasets and their source, protocol, state, support, and optional
expert-goal provenance. It recomputes U targets and repeat variance, verifies
available/censored inventories and feasible B-pair semantics, and derives
descriptor/node inclusion weights from the recorded collector design. It
reproduces the exact capped, hash-ranked, stratum-round-robin node selection,
including root/expert certainty units, and requires equality between feasible
nodes and nodes represented in feasible pair-layout rows before reporting
effective sample size.

Implementer commits: `e7919370f053afaad68d2820aa36f7be131a47d3`,
`e8a3ac0d6d6ba91b33417048dd013e27c4b6bb55`,
`5d84ba676b55857dc8a91d3a976cb97825875d5e`,
`259c86d76e56cc28c81e823795ad6abe30fae5f0`, and
`18faf13c42bb5e5d5168a0b24c4f48726ca14593`.

## Independent reviewer pass

The reviewer found and required fixes for four substantive gaps across review
rounds: repeat/censoring and pair-coverage checks; node weights not tied to
sampling strata and certainty units; selected nodes not reproduced from the
collector's deterministic selection algorithm; and feasible nodes omitted from
the full feasible pair layout. The final exact implementer head
`18faf13c42bb5e5d5168a0b24c4f48726ca14593` was approved after these checks and
adversarial digest-rewrite cases were added.

## Verification and limits

Thirteen focused audit tests passed; Python compilation and `git diff --check`
passed. No GPU, simulator, replay, dashboard, or training run was used. This is
code for auditing future stored artifacts: no dataset, human-authored labels,
independent value oracle, replay evidence, fit, warm-start artifact, or model
outcome is established by this tranche.
