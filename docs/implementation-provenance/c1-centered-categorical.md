# C1 categorical log-probability stability

## Planner

The rich update-18 record showed identical rollout/replay model parameters, raster, globals, hidden state, route/risk costs, candidate feature hash, IDs, and validity mask. Three valid top logits near 1052.5858 varied by one float32 ULP; the selected-action log-probability delta was `0.0001220703125`, above the strict `5e-5` replay guard. The likely fault was precision loss when normalizing large-offset float32 logits.

## Implementation

Commit `8587f1f3c11863384df655b7783d0b928ddf89ac` adds a shared helper that subtracts each row's maximum before constructing `Categorical`. MECO sampling, Plan A/C replay checks, and PPO updates use that helper. The operation is translation invariant and preserves the mask, entropy, and gradients. The strict `atol=5e-5, rtol=5e-5` guard is unchanged.

## Review

Independent static reviewer approved the call-site coverage and mathematical equivalence. It confirmed the diagnostic inputs/model matched and that the observed failure was consistent with common-offset normalization precision loss.

## Limits

The C1 fix's post-change training diagnostic is in progress. Centering does not remove model output jitter, and broader seeded simulator evaluation/replay is still pending.
