# MECO validation campaign planner: three-pass summary

## Planner pass

The audit found that the frozen protocol already enumerates 12 registered
scenarios and 28 distinct validation scenario/stratum jobs, but the matrix CLI
runs one case at a time. A reproducible validation-only planner can assemble
the exact cases without running the simulator. The protocol is currently
provisional, so generated plans must not claim qualification; test seeds must
remain unexported.

## Implementer pass

The planner compiles the deterministic full validation case table and groups
categories only when their exact seed lists match within a scenario and
stratum. It emits 28 validation jobs with protocol-file-bound exact seed
manifests, the default condition/policy/category/seed coverage digests, and a
small replay selector. Optional inputs bind all-five checkpoints, source
identity, and training-configuration hashes. Qualification claims require a
calibrated v2 protocol with frozen safety gates. The implementation rejects
missing, duplicate, malformed, modified, or extra validation cases before
writing; no final-test seeds are serialized.

Implementer commits: `cb4fe69cd4c025a8adfc46e3e290d73e277d70c6` and
`4a5ef0d71b4a5db218deb6f7b85ca90cd86c6ad0`. Integrated candidate commits:
`a1d615dea2465cadad8fcdc8ada63890691cf80f` and
`9cba0a9af8312de28c39a3e2a842364f185165e0`.

## Independent reviewer pass

The reviewer rejected the initial implementation after demonstrating that
removing a complete validation stratum produced a 27-job plan. After exact
case-table validation was added, the reviewer approved final implementer HEAD
`4a5ef0d71b4a5db218deb6f7b85ca90cd86c6ad0`. Review confirmed 28 jobs and no
final-test seed values in the serialized plan.

## Verification and limits

Six focused tests passed, including omitted and extra stratum mutations;
Python compilation and `git diff --check` passed. A CLI dry run generated a
28-job validation plan and refused the current provisional protocol's
qualification claim. No matrix, simulator, GPU, dashboard, or final-test
ledger operation was run. This tranche plans validation inputs only; it has
no evaluation outcomes, replay evidence, qualification result, campaign
runner, or cross-scenario result aggregation.
