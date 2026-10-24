# Runtime hipify source identity

## Planner pass

Training workers recompute the frozen MECO source identity during rollout.
The first native collision-kernel import lets ROCm hipify reformat the tracked
derived file `frc_defense/csrc/tensor_collision_multi_hip_kernel.hip`, so the
identity changes and workers stop before checkpoint writes. Exclude this
generated derivative by exact path, while continuing to hash its maintained
`.cu` and `.cpp` inputs.

## Implementer pass

Implemented in `frc_defense/curriculum_training.py` at commit
`88a71aad0bbbb3b88574a025851a4617eb118c7d`, integrated into `main` as
`2cf6c2b63e7ad444566cca1a3c148ce451d97f52`. `source_identity()` now excludes
the runtime-rewritten HIP derivative and still includes the maintained CUDA
kernel and C++ wrapper. Static compilation and `git diff --check` passed; no
tests or simulator runs were performed.

## Reviewer pass

Independent review approved the exact amended implementation commit and
confirmed the exclusion is limited to the generated file. The `.cu` and `.cpp`
inputs remain covered by the source suffix allowlist.

## Limits

This addresses the source-snapshot failure seen in the first two-worker
diagnostic attempt, which wrote zero optimizer updates. It does not show that
training now succeeds; a fresh source-bound manifest and fresh campaign are
required, followed by live worker and GPU verification. The failed campaign
root is retained for audit and must not be reused.
