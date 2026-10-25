# GPU gamepiece physics: fixed-mask and qualifying HIP implementation

## Three-pass binding

- **Pass 1 plan:** `docs/plans/gpu-simulator-gamepiece-physics-tranche-20261026.md`, planner commit `027f694c4c729c313f48440475907164d985398c`, SHA-256 `4fad637c060cc55478c2999564162c269668b5ab24c5a6c04d2a0aad1ad0ec5b`.
- **Pass 2 implementation:** exact candidate and integrated head `4c604e0f30324d0b20bb38fe5f9b177264a5adcd`, based on `781ab4493ed11949ca861658f94469eaf21b371d` and pushed to `main`.
- **Pass 3 review:** independent static review approved the exact candidate against the plan above. The persisted review summary is [`gpu-gamepiece-fixed-mask-reset-pass3-review-20261026.md`](gpu-gamepiece-fixed-mask-reset-pass3-review-20261026.md).

## Implemented scope

The qualifying path now pins HIP and its supported solver/broadphase configuration, rejects unsupported or mutated configuration, requires graph capture, and fails closed if graph capture fails. Adaptive substep stability bounds are mandatory in qualifying mode; fixed single-step internals are restricted to an active qualifying graph capture. HIP grid rebuild clears all per-world cells before binning. Qualifying reset, bind, advance, substep selection, launch, external commit, spawn separation, and outpost release validate devices, shapes, and masks before state mutation. Fixed-shape masks cover score/pass/launch, reset/preload/random placement, and outpost mutation. Score recording that would materialize device values as Python lists/scalars is rejected for GPU qualification, with failure before the physics evaluation starts. The defense gamepiece wrapper flushes passive score telemetry after gameplay decisions.

The reviewed source path is limited to `frc_defense/_tensor_3v3_gamepieces.py`, `frc_defense/csrc/fuel_physics_hip_kernel.cu`, `frc_defense/fuel_physics.py`, `frc_defense/fuel_physics_runtime.py`, `frc_defense/gamepiece_actions.py`, `frc_defense/meco_track_training.py`, and `frc_defense/tensor_defense_gamepieces.py`.

## Evidence and limits

Static Python compilation and `git diff --check` passed. The exact-head reviewer approved the broadphase rebuild, graph-capture failure behavior, adaptive bound, device and shape guards, fixed-mask gameplay/reset paths, and guarded score-recorder boundary. No tests, HIP compilation, runtime, GPU, simulator evaluation, dashboard replay, or qualification run was performed. This approval is source review only; it does not establish physical WX9100 execution, simulator behavior, score-ledger execution, replay provenance, qualification, or whole-system GPU-only compliance. Qualification currently rejects score-ledger recorder use because its event extraction still requires CPU materialization; that is an explicit fail-closed limitation pending a future GPU-resident event-buffer tranche.
