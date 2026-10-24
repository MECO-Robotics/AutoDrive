# Phase 2 Plan A matched rollout U oracle

## Planner gap

The existing Plan A goal and legal-node supervision paths did not identify a
versioned independent rollout-return U oracle or bind each U target to its
exact matched branch evidence. Existing reports also risked conflating rollout
value estimates with expert choice or globally optimal goal labels.

## Implementation

Commit `88b0bb9270ad912fb0cc651c33990e57c111b9b0` adds the explicit
`matched-prefix-equal-feasible-behavior-gross-return-v1` U contract. It binds
each sampled-node target to the support digest, complete feasible-behavior
inventory, sampled descriptor identities, paired repeat seeds, statuses, and
gross horizon returns. The offline audit verifies the deterministic common
random-number schedule, complete feasible support for available U labels, and
the evidence digest. The fitter records the U semantics and keeps P at P0
because this dataset supplies no independent P labels.

Changed paths:

- `frc_defense/meco_counterfactual.py`
- `frc_defense/meco_plan_a_oracle_audit.py`
- `frc_defense/meco_supervised.py`
- `tests/test_meco_plan_a_oracle_audit.py`
- `docs/MECO_SUPERVISED_WARMSTART.md`

## Review and checks

The independent reviewer approved the exact implementation commit
`88b0bb9270ad912fb0cc651c33990e57c111b9b0`. The focused offline audit suite
passed 17 tests. Python compilation and `git diff --check` passed.

## Limits

No labels, rollout dataset, fitted weights, or warm-start artifact were
generated. No simulator, replay, GPU training, or qualification run was
performed. These labels estimate returns for bounded sampled legal nodes under
the fixed teacher continuation; they do not establish a globally optimal goal
or policy. P remains at P0 until independent P labels exist.
