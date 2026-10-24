# D0 `pickup_simple` calibration audit

## Scope and pinned evidence

This packet audits the existing D0 deterministic-baseline calibration report;
it does not add calibration data or qualify a learned policy.

| Item | Value |
|---|---|
| Report | `qualifications/meco/d0-calibration-pickup_simple.json` |
| Report SHA-256 | `eaf39d5df985c854b157f4abf9fc48a66517cd4864d4486006d0681dda7f76a6` |
| Bundle | `qualifications/meco/d0-calibration-pickup_simple-bundle.json` |
| Bundle SHA-256 | `0ba1f570c352d7167ab94c4f84f22706865c6e5da38afdc77a160a2c85087b5a` |
| Calibration ID | `1be58a36b643f3ed48fc661e0d7454007b8f0f5fb2eb5f9a9bf380c9c4c48030` |
| Source identity SHA-256 | `31e5f85370c3ad83d0b68392de59ec3b525828d2045e86b98aedf309cd8f1350` |
| Policy | `D0_production_deterministic` |
| Scenario and split | `pickup_simple`, `calibration` |

The report contains 1,200 records: four strata × 100 unique seeds × three
independent repeats of the same seed. The four strata are `single` and
`cluster`, each across two shifted-structure layouts. This is calibration
coverage for `pickup_simple` only.

## Uncertainty summary

The reported standard error is computed per stratum as

```text
sqrt(between_seed_variance / n
     + mean_within_seed_repeat_variance / (n * repetition_count))
```

Here `n = 100` and `repetition_count = 3`. Mean within-seed repeat variance is
zero in every stratum, so the reported uncertainty is driven by between-seed
variance. The maximum single-stratum standard errors are:

| Category | Maximum stratum SE | Stratum |
|---|---:|---|
| `single` | `1.814612583750357` | shifted-structures v2 |
| `cluster` | `0.2149605199139075` | shifted-structures v2 |

These figures describe this calibration sample. They do not select operational
margins or establish a policy acceptance threshold.

## Curriculum dependency-closure coverage

Closure size below counts the target scenario and all of its transitive
prerequisites. Only `pickup_simple` has D0 calibration evidence in this report.
For every other target, all listed closure scenarios other than
`pickup_simple` remain without this calibration evidence; the target itself is
included in the missing list.

| Target | Closure size | Missing prerequisite/scenario coverage |
|---|---:|---|
| `pickup_scattered` | 2 | `pickup_scattered` |
| `pickup_multi_zone` | 3 | `pickup_scattered`, `pickup_multi_zone` |
| `active_carry_to_zone` | 4 | `pickup_scattered`, `pickup_multi_zone`, `active_carry_to_zone` |
| `active_shoot_preloaded` | 2 | `active_shoot_preloaded` |
| `inactive_ferry_preloaded` | 2 | `inactive_ferry_preloaded` |
| `active_collect_carry_score` | 6 | `pickup_scattered`, `pickup_multi_zone`, `active_carry_to_zone`, `active_shoot_preloaded`, `active_collect_carry_score` |
| `inactive_collect_ferry` | 5 | `pickup_scattered`, `pickup_multi_zone`, `inactive_ferry_preloaded`, `inactive_collect_ferry` |
| `hub_transition_cycles` | 9 | `pickup_scattered`, `pickup_multi_zone`, `active_carry_to_zone`, `active_shoot_preloaded`, `inactive_ferry_preloaded`, `active_collect_carry_score`, `inactive_collect_ferry`, `hub_transition_cycles` |
| `offense_vs_scripted_pressure` | 10 | `pickup_scattered`, `pickup_multi_zone`, `active_carry_to_zone`, `active_shoot_preloaded`, `inactive_ferry_preloaded`, `active_collect_carry_score`, `inactive_collect_ferry`, `hub_transition_cycles`, `offense_vs_scripted_pressure` |
| `team_delivery_cycles` | 11 | `pickup_scattered`, `pickup_multi_zone`, `active_carry_to_zone`, `active_shoot_preloaded`, `inactive_ferry_preloaded`, `active_collect_carry_score`, `inactive_collect_ferry`, `hub_transition_cycles`, `offense_vs_scripted_pressure`, `team_delivery_cycles` |
| `full_match_offense` | 12 | `pickup_scattered`, `pickup_multi_zone`, `active_carry_to_zone`, `active_shoot_preloaded`, `inactive_ferry_preloaded`, `active_collect_carry_score`, `inactive_collect_ferry`, `hub_transition_cycles`, `offense_vs_scripted_pressure`, `team_delivery_cycles`, `full_match_offense` |

This table describes scenario-level dependency closure, not completion of each
scenario's required category cases or curriculum promotion gates.

## Decision boundary

This artifact records D0 calibration evidence only. No operational margins
have been chosen, no learned policy has been qualified, and no v2 threshold
freeze is asserted. Learned-policy held-out evaluation and its acceptance
decision remain separate work.

## Scoped three-pass record

- **Planner:** scoped this packet to existing D0 calibration evidence, variance,
  and curriculum closure coverage; no thresholds or additional calibration runs
  were selected.
- **Implementer:** added this packet and linked it from the execution status
  ledger in commit `712279b561c056fb804e73a1fef4b6d9fa3f1c73` on isolated branch
  `codex/meco-d0-calibration-audit`.
- **Reviewer:** independently recomputed the report and bundle digests,
  calibration ID, row/repeat coverage, standard errors, and all 12 scenario
  dependency closures; approved that exact commit. This review ran no tests,
  simulator evaluation, or training.
