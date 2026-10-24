# D0 calibration campaign orchestration

**Scope:** Plan, resume, and merge the full registered-scenario D0 calibration campaign from immutable per-scenario shards. This is a scoped transcript summary, not evidence that the campaign was run.

## Planner pass

The planner scoped an orchestration layer over the existing shard contract: require the full 12-scenario closure, create a source/runtime-bound campaign manifest, reuse only validated completed shards, run missing shards, and merge results through an atomic validated publication. The default plan describes **139,800 D0 records**. This is planned work only; none of those campaign records are claimed as completed by this tranche.

## Implementer pass

Initial orchestration commit `91f24a1ffe4feef37d70f2cba414676412811be2` added campaign planning, validation, shard execution/resume, and merge. The final implementer head `dbd84d8e67a92db7bacef46043cae4a2bd316e8b` added restart-safe merge and hardened atomic report publication. The integration candidate carries the tranche in commits `448fe6b6b45e33da232f81dd2d26a712474b06d6`, `e696257d73cfc4518307f5b1a33f4a9b1504b98e`, and `08159fa4d9aab1b58db1197afb650f2ba651ef62`. Exact changed paths: `scripts/meco_d0_calibration_campaign.py` and `tests/test_meco_d0_calibration_campaign.py`.

## Reviewer pass

The independent reviewer approved the exact final implementation commit after **8 focused tests**. The approval covers orchestration and artifact-publication contracts only.

## Limits

The 139,800 records are planned, not run. No GPU or simulator results, CI result, or qualification result is claimed. This tranche does not establish D0 campaign completion or learned-policy qualification.
