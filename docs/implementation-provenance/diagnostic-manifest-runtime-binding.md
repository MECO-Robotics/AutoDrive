# Diagnostic manifest runtime binding

## Planner pass

The five-track manifest validator correctly binds protocol bytes, source
identity, and evaluation seed partitions. The launch path then incorrectly
passed that protocol as a runtime validation binding even when the manifest
selected the diagnostic split. The supervisor rejects protocol-bound
diagnostic runs. Preserve offline manifest validation, but omit the runtime
evaluation protocol for diagnostic launches; keep exact protocol binding for
validation launches.

## Implementer pass

Implemented in `frc_defense/meco_supervisor.py` at commit
`7499d5b266dc60727497c763a03249c6d298b8ff`, integrated into `main` as
`6d60294cd3d1741b0c685b15fb2e71e3d14efa0b`. Diagnostic manifests now pass
`evaluation_protocol=None` to the runtime supervisor. The manifest is still
validated first, and validation-split launches still pass the bound protocol.
No tests or simulator runs were part of this scoped fix. `py_compile` and
`git diff --check` passed in the implementer worktree.

## Reviewer pass

Independent review approved the exact implementer commit. The review confirmed
that source, protocol digest/ID/partition checks and diagnostic seed
disjointness remain enforced; validation still resolves its exact case and
seeds, while automatic test selection remains forbidden.

## Limits

This fixes campaign startup wiring only. It does not supply Phase 2 warm-start
artifacts, complete calibration, held-out comparisons, Phase 4 results, or
qualification evidence. Training launch and live GPU use must be verified
separately.
