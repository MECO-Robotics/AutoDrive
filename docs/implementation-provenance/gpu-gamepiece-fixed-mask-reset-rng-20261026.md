# GPU gamepiece fixed-mask/reset checkpoint

**Status:** Pass 2 intermediate implementation checkpoint; not a completed or validated gamepiece-physics tranche.  
**Plan:** `docs/plans/gpu-simulator-gamepiece-physics-tranche-20261026.md` (SHA-256 `4fad637c060cc55478c2999564162c269668b5ab24c5a6c04d2a0aad1ad0ec5b`).  
**Base:** `84982b536199a939160682559c92a63cd737eea7`; implementer branch `codex/gamepiece-physics-impl-20261026`.

## Changes in this checkpoint

- Ferry/pass, score/respawn, and `launch_fuel` no longer use a device-derived `any()` result copied to the host to skip the fixed-shape update path. Ferry destination sampling now runs at fixed shapes for empty and nonempty pass masks.
- `separate_spawned` uses the existing fixed-shape masked implementation for every call; selected-spawn masks control writes on device. External outpost releases use fixed-shape per-world masks, preserve first-stock-slot selection and repeated release ordering, synchronize released slots into 3D physics state, and update contact-cache invalidation state.
- Randomized preload placement no longer boolean-compacts `[world, preload]` writes. Random gamepiece placement no longer copies accepted rows to the host or uses host rejection sampling. It assigns active pieces to a random permutation of fixed, separated grid cells after a static capacity check.
- The RNG contract is explicitly labeled `fuel-gamepiece-rng-fixed-mask-v2` on initialized environments. This version is required because fixed-shape ferry/backer work and grid-cell permutation change the old pass-dependent and host-rejection random draw schedule. Seed parity with the previous implementation is not claimed.

## Source-only checks and limits

Ran `python3 -m py_compile` on the four edited Python modules and `git diff --check`; both passed. No tests, simulator, GPU, dashboard, replay, or campaign commands were run. These checks establish syntax and patch formatting only; they do not establish selection equivalence, placement behavior, RNG fixture parity, physics synchronization in execution, or GPU residency.

Known remaining plan blockers at this checkpoint include host CPU broadphase construction in the Torch reference path, host-colored contact grouping, host sleep-island behavior when that option is selected, adaptive substep selection through a host scalar, and the absence of an audited qualifying-mode gate that prevents `auto` backend fallback to Torch. These remain in scope and require implementation plus a later independent Pass 3 review. No promotion, calibration, qualification, or simulator-behavior claim is made.
