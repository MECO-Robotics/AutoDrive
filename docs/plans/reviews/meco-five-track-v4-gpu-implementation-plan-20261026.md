# Independent Pass 1 review: five-track V4 GPU implementation plan

- **Verdict:** ACCEPT
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** independent Pass 1 plan review assigned by `/root`
- **Plan author:** `plan_a_owner` (planner, not implementer or human approver)
- **Reviewed plan commit:** `223ac4a3f3c657193ef5623b383716364a9395d5`
- **Reviewed file:** `docs/plans/meco-five-track-v4-gpu-implementation-plan-20261026.md`
- **Git blob:** `7f57f33f9afc1ef35eff17a50b4f9d84a351d8bc`
- **Plan file SHA-256:** `42b138445c25dffad7402e70faa16c6fb5a142f43497353aa08741a70e75e6aa`
- **Pinned source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Pinned reconciliation:** commit `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, file SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; exact Pass 3 ACCEPT record commit `5cb1a2d525168ff0443089cf329e66c6ecb917b8`, record SHA-256 `0df700dcaef1ed35672efc468bdcfd78fac829c31bcf7a88b3c51f1951f06c32`.
- **Reconciled profile hash:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`
- **V4 envelope profile digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`

## Review scope and evidence

The plan commit is a direct child of the exact pinned source base, and its sole change is this planning document. I verified that the reconciliation commit is not an ancestor of the source base and that its artifact is absent there. The plan handles this reproducibly: before implementation packages start, it requires a docs-only contract-anchor bridge, exact artifact/profile hash checks, proof that the source tree remains identical to base, and independent review of that bridge. It prohibits importing unrelated reconciliation-branch source history.

The plan correctly separates the accepted reconciliation artifact, canonical reconciled profile hash, and raw V4 envelope `profile_digest`. The hashes match the exact artifact and its accepted review. It defines a coherent staged path with gates and acceptance evidence: G0 protects the active diagnostic campaign and leases; P0 establishes immutable score/rules, seed partitions, calibration, margins, protocol v5, approvals, and the two-device supervisor configuration; P1 covers typed schema, device-side score events and artifacts; P2 covers full-path device residency; P3 covers A1/A2/A3; P4 covers C1/C2; P5 covers PPO, end causes, RNG and atomic checkpoints; and P6 covers five-lane training, read-only evaluation and qualification. Each package must get its own finite allowlist, named distinct planner/implementer/reviewer, exact base and Pass 1 plan before it starts. The shared trainer has one package owner, integration occurs in dependency order, and each changed integration head is reviewed anew.

The operating boundary is clear and fits the active campaign constraint. The plan explicitly prohibits touching campaign PID `2750253`, its dashboard, workers, checkpoints, output, leases or devices until the user manually stops it and both GPU leases are independently confirmed released. G0 additionally requires separate authorization before starting the new implementation campaign. No work is claimed already complete. Training after those gates is to continue until the user's manual stop; evaluations use immutable snapshots, declare their GPU share and cannot mutate trainer state.

GPU-only scope is explicit for algorithmic training, simulation, scoring, candidate/route work, PPO, RNG, evaluation, calibration and qualification on the two assigned WX9100 devices. CPU numerical fallback, CPU simulation/scoring and silent migration are disallowed, and unavailable devices cause a stop/replan gate rather than P600/CPU fallback. The plan restricts CPU to supervision, configuration, I/O and noncomputing control, and requires UUID/runtime pinning, full-path transfer/synchronization evidence, per-kernel placement/memory records and evaluation-throughput accounting.

All five tracks are named consistently as A1, A2, A3, C1 and C2, with their implementation packages, shared profile/task/seed controls, per-track checkpoints and failure isolation. P0 freezes disjoint seed partitions and calibration-derived margins before candidate results. P6 requires the frozen target-scoped v5 protocol, bound checkpoint/source artifacts, all applicable category/layout/defender/safety/paired-control/score/progress/confidence gates, and three consecutive passing Stage 1 rounds for each track. It excludes legacy protocols, diagnostic chunks, prior 100/100 reports and partial updates as qualification evidence; a failed case or changed protocol/checkpoint resets the affected streak. Final-test isolation is explicit.

The validation plan covers the project AGENTS.md requirements: every behavior test that exercises simulator/game behavior is paired with a representative seeded end-to-end simulator evaluation, and the scenario, seed, fresh replay ID, provenance and result are recorded. Before any dashboard replay is generated or reviewed, serving services must be restarted after code changes; backend and frontend must be verified from the current integration checkout, services active and endpoints responsive; a fresh replay must then be generated. Stale replays are marked outdated. The plan requires reporting tests, simulator evaluations, replay/scenario IDs, seeds and dashboard currentness. These are future scoped validation gates, not activity performed by this plan review.

The plan is complete for the requested planning scope: it addresses all five approaches and Stage 1 qualification while preserving implementation, campaign, device, dashboard and qualification gates. It does not claim source conformance, a completed qualification result, human approval or ABI publication. No blocking Pass 1 finding remains.

## Review limits

This ACCEPT applies only to the exact planning artifact and authorizes only progression through the plan's stated review gates. It does not authorize implementation, tests, builds, device/runtime work, training, evaluation, dashboard actions, ABI publication or qualification. No tests or runtime actions were performed during review.
