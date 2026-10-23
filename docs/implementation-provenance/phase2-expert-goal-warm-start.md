# Plan A expert-goal warm-start intake and fitter

## Planner

The exact plan file and candidate copy match SHA-256 `718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10`. The existing expert-goal source is a deterministic feasible route-cost demonstration, not an expert or value oracle. It marks U unavailable. The loader already enforces replay-bound goal, action, behavior, support, censoring, protocol, and source identity.

## Implementation

Commit `15fa367fd5975a0fc7af641090a46a9ca89284bd` adds `warm-start-plan-a-expert-goals-v1` for A1/A2/A3, a weights-only artifact validator/loader path, indexed-CUDA checks, and an optional replay-bound selected-goal certainty unit in legal-node collection. The fitter trains B only over available feasible behaviors at the annotated goal, accepts U only for finite complete uncensored values, and resets P to p0. It does not infer expert optimality or a P target.

## Review

Independent reviewer approved the implementation as an annotation intake and fitter. Review covered source/support replay binding, U/B masks, A1 versus A2/A3 node indexing, p0, artifact validation/loading, supervisor validation, and indexed CUDA requirements.

## Limits

There is no producer for the annotation JSONL in the repository; annotation rows and the authority file must be supplied externally. The deterministic goal source is not an independent U oracle. No annotation dataset has been collected, no artifact fitted, and no simulator validation run. This tranche does not establish complete Plan A Phase 2 evidence.
