# Independent Pass 1 review: scoped P0 Phase 0 package plan

- **Verdict:** REVISE
- **Review date:** 2026-10-26
- **Reviewer identity:** `/root/closure_wave1_reviewer`
- **Reviewer task:** independent Pass 1 package-plan review assigned by `/root`
- **Plan author:** `plan_a_owner` (planner only)
- **Reviewed plan commit:** `0872a700e58b671fd2a14381609410903afb1ee8`
- **Reviewed file:** `docs/plans/meco-v4-p0-phase0-scoped-package-plan-20261026.md`
- **Git blob:** `50b0283442eefed3b7dc2ae4e3923fbec218fb40`
- **Plan file SHA-256:** `0097d70e7cb9725d4dc40d05b32ee029ed91a448900efa5874649efc71d4f5ab`
- **Accepted overall plan:** commit `5d02eab99738cefa0318d4af4018c0450ecca685`, file SHA-256 `c7e6d12c4c9e6c9d48d05015e0eb8eac91fe9b42c26e49a52fb81e52c1a4a37c`; independent Pass 1 ACCEPT record commit `9cf9aad57`, record SHA-256 `ddef134429f39365d254d490ea930c03fb6db963dd3f986c5c4e7579e951bc3d`.
- **Implementation source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted reconciliation:** commit `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`, artifact SHA-256 `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`; accepted review record commit `5cb1a2d525168ff0443089cf329e66c6ecb917b8`, record SHA-256 `0df700dcaef1ed35672efc468bdcfd78fac829c31bcf7a88b3c51f1951f06c32`.
- **Reconciled profile hash:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`
- **V4 envelope digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`

## Review scope and evidence

The package plan pins the accepted overall-plan amendment and exact reconciliation/profile hashes. Its proposed content correctly distinguishes the accepted `simulated_match_score` estimand from curriculum reward and official match points, identifies observed producer/consumer evidence while deferring exact confirmation to P1's source map, notes the absent score-ledger path, and distinguishes current v4 qualification behavior from the required v5 contract. It defines explicit target, seed split, calibration-method, uncertainty, margin, identity, fail-closed, and human approval decisions. Its G0 boundary protects the active campaign, dashboard, workers, outputs, devices, and leases; implementation requires manual stop and independent lease-release evidence. The plan retains distinct planner, implementer, and reviewer roles and includes the AGENTS.md seeded simulator/replay and current-dashboard requirements as future validation gates.

## Blocking finding: package workflow contradicts the accepted P0a/P0b boundary

The accepted overall plan defines P0a as a documentation/configuration decision package that precedes source implementation and produces no final artifact or protocol instance. P1–P6 source implementation may start only after P0a is accepted. P0b then follows the reviewed P1–P6 source freeze and performs final calibration/artifact creation; P6 runtime qualification follows P0b.

This scoped plan does not preserve that boundary. Its Pass 2 start criteria require “P0a decisions and required human decisions” to already be accepted, while its Pass 2 cadence says the implementer “produces P0a decisions.” That leaves no defined authorized work phase that can produce and review P0a before its acceptance is required. The plan also puts `frc_defense/meco_qualification.py`, the scheduler, qualification documentation, and tests in one P0 Pass 2 allowlist, calls P0a source implementation in the cadence, and makes P0 the owner of v5 parser/schema/source/device binding and calibration runner code. Those source changes are incompatible with the top-level P0a source-free gate. If delayed to P0b, they would need to be frozen and reviewed before P0b calibration and included in the final bound tree; the current dependency/ownership plan does not define that integration and freeze ordering.

Revise the scoped workflow to define a P0a decision-authoring and exact-head review stage that can occur after the applicable plan/G0 gates without source edits, then enable P1–P6 source work only after P0a ACCEPT. Place any v5 runner/schema source needed for qualification in a pre-P0b source package with explicit ownership and integration dependencies (or assign it within P1–P6), so it is frozen and reviewed before P0b binds the source tree. Keep P0b limited to calibration and immutable output generation/verification after that freeze, unless the accepted top-level plan is amended to expressly include later P0b-owned source work and define when that source freezes before calibration. Resolve the current contradictory Pass 2 start and delivery language and request a fresh exact-head review.

Because the current task boundaries leave P0a impossible to complete under its own gates and leave P0 source ownership ambiguous against the required source freeze, this exact-head plan is **REVISE**. No code, tests, runtime, GPU, dashboard, or campaign action is authorized by this review.

## Review limits

This REVISE applies only to the exact package plan pinned above. It does not authorize implementation, tests, builds, calibration, device/runtime work, dashboard actions, training, evaluation, ABI publication, or qualification. No source or runtime actions were performed during review.
