# MECO five-track V4 GPU implementation and qualification plan

**Date:** 2026-10-26  
**Status:** Planner-authored candidate; **pending independent Pass 1 review**  
**Delegated planner identity:** `plan_a_owner` (not an implementer or human approver)  
**Base implementation checkout:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`  
**Reconciliation commit:** `03bd0a7a3ef4ea0d2506ec4f9b031745ba53ada9`  
**Reconciliation artifact SHA-256:** `294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`  
**Canonical reconciled profile hash:** `5ec637676c0acf02f9769b1629860122c6c5332914acdcfbb9c04adaafea70e3`  
**V4 envelope profile digest:** `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`

## Purpose and authority

This plan defines an ordered, reviewable route from the current five-track
implementation to the reconciled MECO V4 contract, GPU-only algorithmic
training/evaluation on the two WX9100 devices, and five-track Stage 1
qualification. It covers A1, A2, A3, C1 and C2. It does not assert that the
current source conforms, that any track has passed Stage 1, or that current
diagnostic results are qualification evidence.

This file is the sole output of this planning tranche. It is not source or
schema implementation, a test or runtime instruction, training authorization,
ABI publication, or human maintainer approval. After an independent Pass 1
ACCEPT, implementation still proceeds only through the scoped Pass 2 and exact
head Pass 3 gates in this plan. A Pass 1 REVISE requires updating this plan and
reviewing the new exact head before any implementation package starts.

The immutable contract anchor is the reconciled artifact and profile hash
above. Implementers must consume the canonical profile declaration in that
artifact byte-for-byte. In particular, preserve its five track codes, fixed
profile dimensions, Plan A and Plan C capacities/order, typed score-event
table, PPO/RNG registries, 314,469,051-byte payload and 96-byte envelope.
Any required change to a reconciled field, digest, encoding, status code,
capacity, RNG registry, or byte total stops the affected work package. It
requires a separately reviewed replacement/reconciliation and a new exact
profile hash before code continues.

## Hard operating boundary

The live diagnostic campaign PID `2750253` is outside this plan's authority.
Do not stop, signal, reconfigure, evaluate through, restart, or otherwise touch
that process, its dashboard, workers, checkpoints, output directory, GPU
leases, or devices. Before the G0 execution gate is satisfied, only isolated,
docs-authorized source and test-file authoring plus static source/diff review
may proceed after the applicable package plan and prerequisites are accepted.
This does not authorize build or compile, test execution, numerical work,
simulator, calibration, evaluation, dashboard operation, device or GPU use,
training, checkpoint mutation, integration, merge, or promotion. None of those
operations may begin until the user manually stops the campaign and both
WX9100 leases are independently confirmed released. The plan author did not
perform any of those operations.

After those release conditions and plan acceptance, all numerical and
algorithmic computation in training, simulation, scoring, candidate/route
processing, PPO, RNG, evaluation, calibration, and qualification must run on
the two assigned AMD WX9100 devices. CPU numerical fallback, CPU simulation,
CPU scoring, and silent device migration are disallowed. CPU use is limited to
process supervision, configuration, filesystem/network I/O, and control
messages that do not compute policy, physics, reward, score, metrics, or
qualification results. Device UUID/runtime identity and per-process placement
must be pinned and recorded. If both WX9100 devices are unavailable or cannot
execute a required kernel, stop at the readiness gate and revise the device
plan; do not fall back to a P600 or CPU.

The eventual five-track campaign must start from fresh, source-bound
checkpoints after implementation changes. It runs continuously until the user
manually stops it. Read-only evaluations may run on immutable checkpoint
snapshots without stopping or reconfiguring the training workers; their GPU
time share and impact on update throughput must be declared in the frozen
supervisor configuration. Stage 1 qualification reports must not mutate
training checkpoints, curriculum state, or promotion state until the entire
round and immutable report are committed.

## Reconciled product and profile decisions

The accepted reconciliation resolves the following target behavior:

- Five learned policies: A1/A2/A3 use Plan A dense legal-pose utility and
  nonnegative traversal cost with behavior logits; C1/C2 score the same
  bounded, deterministic candidate contract, with C2 adding map context.
- Profile: `N=1`, `S=6`, `R=32`, `T=12`, `E=504`, `K=16`, raster
  `[6,25,51]` plus seven globals in the pinned order. Plan A uses 12,096
  padded nodes, 96,768 support rows, ten edges per padded node and 14 ancestor
  levels; `node_valid` defines the live subset. Plan C uses 537 source rows,
  feature widths 20/40, 9,681 route rows, 152 chunks of at most 64 and 73 path
  points.
- Common command identity, sampled action and old log probability remain
  linked; executed per-tick actions remain a separate trace. Terminal,
  truncation, bootstrap, actor/critic/row/command validity and reset semantics
  follow the adopted PPO/RNG row. Failed validation, digest, capacity, schema,
  replay or RNG checks leave model, optimizer and PPO RNG state unchanged.
- The fixed score-event table is 2,304 records × 59 bytes. Reserve and stage
  the full event capacity before simulator mutation. Invalid unused rows are
  zeroed except `valid=0`; malformed events or overflow fail the whole
  transition before mutation.
- The evaluation estimand is `simulated_match_score`: accepted active-HUB
  FUEL awards only, not official match points. Use the fixed strategic horizon
  of 160 physics ticks, at most 12 ticks per macro (the last macro may be
  partial), gamma 0.993 per macro with branch exponent zero and one increment
  for each continuation decision. Keep route and risk costs separate from
  gross reward.

These are target contract facts from the accepted reconciliation, not claims
about current runtime behavior. In particular, the reconciliation identifies
the current single `done=ended or timeout` GAE/bootstrap path, missing typed
end cause in counterfactual collection, lack of an immutable Phase 0
score/rules-bound artifact, absence of a target-scoped v5 protocol, and lack
of five bound checkpoint manifests as open implementation/qualification work.

## Three-pass ownership and worktree protocol

Every code/schema package uses three distinct roles and exact-head artifacts:

1. **Pass 1 planner:** owns the package plan and its finite file allowlist,
   prerequisites, interface, acceptance evidence, and risk gates. The planner
   does not implement or review its own package.
2. **Pass 2 implementer:** works in a new ZFS-backed worktree and branch from
   the exact approved integration base. Before G0, the implementer may author
   only files in the accepted package allowlist and perform static source/diff
   review; no build, compilation, test execution, numerical/runtime work,
   device use, or integration is allowed. Record and push each reviewable
   source-only slice. After G0, package validation may begin under its
   accepted plan.
3. **Pass 3 reviewer:** a different agent first performs a source-scope and
   interface review of the exact source-only candidate head. The reviewer may
   record only SOURCE/API ACCEPT or REVISE. SOURCE/API ACCEPT pins a candidate
   API for dependent authors, but is not package-final Pass 3 ACCEPT and does
   not permit dependent packages to integrate or runtime-use the API. After
   G0, the reviewer checks exact hashes, tests, device-residency
   evidence, and required simulator/dashboard evidence; only the final exact-
   head ACCEPT permits integration. REVISE requires a corrected commit and a
   new review record. Authors do not review their own work.

Worktree paths below are proposed unique locations under the repository's
ZFS-backed `.worktrees/` parent. Branch names include the package ID and date.
Create them after the applicable package plan and dependencies are accepted,
in fresh ZFS-backed worktrees that do not use the live campaign checkout or
output directory. G0 is required before any build, test execution, simulator,
calibration, evaluation, dashboard, device/GPU use, integration, merge or
promotion.

| Package | Planner | Pass 2 worktree / branch | Pass 3 reviewer | Main owned scope |
|---|---|---|---|---|
| P0a — Phase 0 preregistration | phase0 planner | `.worktrees/autodrive-v4-p0-phase0` / `codex/v4-p0-phase0-YYYYMMDD` | independent protocol reviewer | documentation-only candidate score/rules, v5 requirements, cases/seeds and calibration/margin method; human decisions precede P0a Pass 3 ACCEPT |
| P0b — Phase 0 final calibration | phase0 planner/operator | fresh exact-base worktree after reviewed source freeze | independent protocol reviewer | post-P1–P6 runtime calibration and generated immutable Phase 0 artifact/protocol only; no qualification source edits |
| P1 — V4 schema, score events and envelope | ABI/artifact planner | `.worktrees/autodrive-v4-p1-schema` / `codex/v4-p1-schema-YYYYMMDD` | independent schema reviewer | schema and serialization modules, canonical score event producer/consumer paths after discovery, artifact/manifest code and schema fixtures |
| P2 — GPU-resident environment/observation/route core | device-runtime planner | `.worktrees/autodrive-v4-p2-device-core` / `codex/v4-p2-device-core-YYYYMMDD` | independent residency reviewer | environment observation, target handles, candidate generation, route/search kernels and their dedicated modules |
| P3 — Plan A A1/A2/A3 | Plan A planner | `.worktrees/autodrive-v4-p3-plan-a` / `codex/v4-p3-plan-a-YYYYMMDD` | independent Plan A reviewer | Plan A support, model adapter and A1/A2/A3-specific kernels/fixtures |
| P4 — Plan C C1/C2 | Plan C planner | `.worktrees/autodrive-v4-p4-plan-c` / `codex/v4-p4-plan-c-YYYYMMDD` | independent Plan C reviewer | candidate feature/model paths and C1/C2-specific kernels/fixtures |
| P5 — PPO, RNG, end causes and atomic checkpoints | PPO planner | `.worktrees/autodrive-v4-p5-ppo` / `codex/v4-p5-ppo-YYYYMMDD` | independent PPO/RNG reviewer | sole owner of shared `frc_defense/meco_track_training.py` edits, PPO buffer/update, Philox implementation and update/checkpoint tests |
| P6 — Five-track supervisor and qualification reports | qualification planner | `.worktrees/autodrive-v4-p6-qualification` / `codex/v4-p6-qualification-YYYYMMDD` | independent qualification reviewer | v5 schema/runner, five-lane scheduling, immutable evaluation, promotion and report/dashboard data paths; owns `meco_qualification.py` and its qualification tests |

Planner and reviewer identities are assignment gates, not pre-filled consent.
No package starts without named distinct agents, a Pass 1 accepted package
plan, base commit, exact path allowlist, and accepted dependency heads. Keep
shared high-conflict files out of parallel packages: P5 alone edits the shared
trainer; other packages provide reviewed APIs and tests around that integration
boundary. A merge queue integrates packages in dependency order. Each
integration commit is reviewed as a new exact head before dependent packages
rebase. No output data, checkpoints, qualification state, generated binaries,
or active campaign files enter source commits.

## Work packages, dependencies, and acceptance gates

### Gate G0 — runtime, device and integration release

**Prerequisites:** independent Pass 1 ACCEPT of this exact plan; the user has
manually stopped diagnostic campaign PID `2750253`; GPU leases have been
independently released; and the user or repository maintainer has explicitly
authorized the new implementation campaign. Keep these as recorded gate
evidence, not assumptions.

The pinned reconciliation commit is on a history that diverges from the
requested implementation base `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`;
its artifact is not present in that source checkout. After Pass 1 and before
source/schema packages start, create a reviewed, docs-only contract-anchor
integration commit that adds the exact reconciliation artifact bytes at its
pinned path. Verify the artifact SHA-256 remains
`294a02012427c2ceeb638ea0cd337d1fa7908cf47fb9d0f07bef29ed6790510c`, the
canonical profile hash is unchanged, and the implementation source tree is
identical to base `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`. Do not merge the
reconciliation branch wholesale or import its unrelated source history. A
distinct reviewer accepts this exact bridge head before it becomes the base
for implementation worktrees.

**Acceptance:** accepted plan commit/hash, campaign stop evidence from the
authorized operator, separate lease-release evidence, named package
owners/reviewers, reviewed contract-anchor bridge commit, and exact integration
base. Until G0 is complete, isolated source/test authoring and static review
may proceed only under the accepted package plans and prerequisites. No build
or compile, test execution, simulator/numerical work, device/GPU use,
dashboard, calibration, evaluation, qualification, training, checkpoint
mutation, integration, merge or promotion may begin. This plan does not claim
G0 is satisfied.

### P0a — preregister Phase 0 rules and protocol before source implementation

**Dependencies:** accepted P0 package plan and named distinct owner/reviewer.
G0 is not required to author or review this docs-only candidate. This is a
documentation/configuration decision package;
it precedes P1–P6 source implementation. It does not run calibration or emit
the final Phase 0 artifact or protocol.

1. Propose in the candidate the `simulated_match_score` estimand and
   accepted-event rules:
   accepted active-HUB FUEL awards only. Current source evidence identifies
   `fuel_scored_event` and `fuel_score_count` in
   `frc_defense/gamepiece_actions.py`, consumed by reward/info paths in
   `frc_defense/tensor_3v3.py`; `frc_defense/curriculum_metrics.py` is a
   distinct task-reward metric. P1 owns implementation of the reconciled
   typed event table. P0a records the intended semantics and requires P1 to
   verify producer/consumer sites against its exact source base.
2. Specify the proposed immutable target-scoped protocol v5 schema, including target
   IDs, transitive prerequisite closure, case/layout/opponent table, field and
   rule identity, split labels, metric definitions, stopping/promotion rules,
   artifact provenance and fail-closed behavior. Existing v1/v2 protocols
   remain history; they do not satisfy the v5 contract.
3. Propose disjoint train, calibration, validation and final-test case/seed
   partitions and their canonical bytes before candidate outcomes are
   observed. Record the case-generation procedure, seed derivation, split
   membership and permitted regeneration conditions. Any change to source,
   rules, case matrix or seed-generation code requires a new preregistration
   revision before results are examined.
4. Propose the D0 calibration design, repeats, uncertainty calculation,
   useful-gain and safety noninferiority margin method, minimum evidence and
   decision procedure. Do not finalize numeric margins from unrun or
   unreviewed evidence, and do not tune the method after candidate results.
   Record which product/maintainer decisions require explicit human approval;
   hashes and agent-authored records are not approval or authentication.
5. Propose source-identity inputs and the canonical Phase 0 artifact/protocol
   hash boundary. Specify that P0b binds the final reviewed source tree,
   reconciled profile, device pair, score rules, exact cases/seeds, calibration
   evidence, margins and approval record. P0b must fail closed if any bound
   input changes.

**P0a delivery and acceptance:** Pass 2 produces a documentation-only
candidate preregistration containing the proposed schema requirements,
estimand/rules, cases/seeds and statistical method. The authorized maintainer
resolves all required product/rules decisions against that candidate before
Pass 3 review. Pass 3 ACCEPT applies only to the exact candidate plus recorded
decisions; it unlocks P1–P6 source-only authoring when each package plan and
its other dependencies are accepted. G0 remains mandatory before package
validation, integration, merge or promotion. P0a has no
source or test allowlist and produces no calibration results, final Phase 0
artifact, protocol instance, training or qualification result.

### P0b — runtime calibration and generated artifacts after source freeze

**Dependencies:** accepted P0a; P1–P6 source implementation and integration
are complete, frozen and independently reviewed, including all scoring,
simulation, evaluation and qualification-runner source owned by P6. P0b does
not depend on P6 runtime qualification results. P6 runtime/qualification does
depend on accepted P0b. This sequencing prevents a P0↔P6 cycle.

1. Recompute the source/rules/config identity from the exact frozen
   integration commit. Verify the final score producer/consumer mapping,
   field geometry/legal graph, robot/physics/sensor/controller settings,
   case matrix and source-bound metric code against P0a. If any relevant
   source or preregistered input changed, return to P0a revision or restart
   calibration as the change requires.
2. Run only the preregistered D0 cases, seed partitions and repeat schedule on
   the two assigned WX9100 devices, with no CPU numerical fallback. Retain
   raw records and independently recompute the prescribed uncertainty and
   margins. Existing qualification code's older protocol versions and
   single-indexed-device invocation do not demonstrate the v5 contract or
   two-WX9100 residency.
3. Emit (without source edits) the immutable Phase 0 artifact containing
   exact source/rules/profile/
   device identity, score estimand/rules, cases and split seed bytes/hashes,
   calibration records, uncertainty, margins and explicit approval
   provenance. Emit the final target-scoped v5 protocol bound to that artifact,
   exact target and prerequisite closure, metrics and promotion predicates.
   Never overwrite legacy protocol or calibration records.
4. Verify artifact and protocol bytes/hashes, source binding, seed disjointness,
   calibration recomputation and required approvals. A post-freeze change to
   any scoring/simulation/evaluation/qualification source invalidates the
   affected P0b source binding; regenerate and review P0b before fresh
   training or qualification.
5. Pin the five-track supervisor config and device identities. Training and
   Stage 1 qualification may begin only after P0b is accepted and fresh
   source-bound checkpoints are available.

**P0b acceptance:** independent exact-head review accepts the generated
artifact and protocol; calibration recomputes from retained records under P0a's method;
the final hashes bind the frozen P1–P6 integration source and reconciled
profile; seed partitions are disjoint; margins and all required human
approvals are explicit. No Stage 1 round uses an uncalibrated, stale or legacy
protocol.

### P1 — V4 typed schema, device-native score events and artifacts

**Source-only authoring dependencies:** accepted P0a preregistration for score
source/rules and exact reconciliation/profile hash. G0 is required before
P1 builds, test execution, device/numerical validation, integration, merge or
promotion. P1 begins with source tracing because the named
score-ledger file is absent from the base checkout.

1. Trace every authoritative event source and sink in the current simulator,
   environment, trainer, counterfactual collector and metric/report path.
   List event code, source site, event identity, pre-mutation validation,
   tick/episode reset and current host/device boundary. Return a source map to
   the planner before choosing the implementation path.
2. Implement the reconciled 2,304 × 59-byte typed event table, fixed event
   codes/source-site identities and device-side capacity reservation/staging.
   Prove event uniqueness/order, valid=0 zero fill, exact quantity/identity,
   per-tick and episode reset rules, and all-or-nothing failure before any
   score or simulator mutation. Keep the 32,768 episode recorder capacity, if
   still used, separate from this 2,304-record batch ABI capacity.
3. Implement V4 row/profile structures and exact little-endian packing with no
   implicit padding; validate every field against the canonical declaration.
   Device computes row/support/transition digests and the complete payload
   digest. Host copies opaque bytes for persistence; host code does not decode,
   hash, repair, select, or gate on dynamic payload values.
4. Implement the v2 96-byte envelope and strict version-first, exact-length,
   overflow-safe, fail-closed reader. Bind source/config/checkpoint/profile,
   code registries, RNG state and five checkpoint manifests. No partial bundle
   is published on schema, digest, capacity, nonfinite-valid-field or event
   errors.

**Acceptance:** byte-by-byte field fixtures and independently recomputed
row/payload/envelope totals equal the reconciled values; overflow, malformed,
stale and digest failure leave no partial event, artifact or simulator
mutation; GPU-produced digest agrees with independent offline reference
fixtures without moving numerical work to CPU. Any schema/profile byte delta
reopens reconciliation.

### P2 — GPU-only environment, observation, candidate and route pipeline

**Dependencies:** P1 interfaces; exact profile and accepted P0a device/runtime
identity preregistration. Final P0b source binding follows the P2 source freeze.

1. Build a residency ledger for every tensor and algorithm in the entire
   train/eval route, including simulator state and physics, observations,
   target tracking, rasterization, legal geometry, candidate construction,
   route costs, reward/score events, PPO, digest, metrics, curriculum and
   safety checks. For each, name producer, consumer, device, dtype/shape,
   kernel, permitted transfer boundary and failure behavior.
2. Convert current scalar reads/list conversions and host loops into
   device-side tensor operations or native HIP kernels where they are on the
   algorithmic path. Known audit hotspots from the pinned base include
   `.cpu()/.tolist()/.numpy()/.item()` in
   `frc_defense/meco_track_training.py`, per-target host pose/velocity and
   `nonzero().tolist()` loops in `meco_candidates.py`, host serialization and
   scalar pulls in `meco_supervised.py`, and CPU state/reward/route material in
   `meco_counterfactual.py`. These are search leads, not a complete audit or
   an assumption that every listed operation is on the hot path. Audit the
   exact call graph and preserve only permitted I/O/control boundary copies.
3. Preserve snapshot identity, 504 target slots and four-int handles,
   validity, reset generation, no-wrap semantics and stale-handle rejection.
   Candidate ordering/masks, Plan C 537-row source table, 16-wide ordinary
   selection plus continuation, route chunk scatter, all 9,681 route sinks,
   path points and Plan A support must stay on device.
4. Implement exact two-WX9100 affinity and orchestration without CPU fallback.
   For all critical kernels, verify both devices execute the expected device
   code and retain per-kernel transfer/synchronization accounting. Provide a
   stable dual-device policy for five single-track jobs (fair GPU time slices
   or explicitly assigned tracks), prevent oversubscription from silently
   pushing work to host, and record throughput impact when evaluation shares
   the devices.

**Acceptance:** full path audit has no unapproved algorithmic CPU operation or
device-to-host read; profiler/trace evidence covers the complete representative
training and evaluation path, not only model inference; both WX9100 devices
show expected kernels and memory ownership; simulation state and score outputs
match seeded reference records; failure is explicit if residency or device
identity breaks. A CPU fallback is a gate failure.

### P3 — Plan A A1/A2/A3 and fixed node support

**Dependencies:** P1 schema, P2 observation/graph pipeline, accepted Plan A
row in the reconciliation. Keep the Plan A owner scope separate from Plan C.

- Implement 12,096 padded nodes with `node_valid` as a strict legal subset;
  96,768 ordered support rows, ten edge slots per padded node, 14 ancestor
  levels and float64 costs exactly as reconciled. Bind field order and node
  validity into support/input digests. Preserve fail-closed empty support and
  A1/A2/A3 action-to-command identity.
- Implement the three architecture heads under identical observation,
  safety, legal action and PPO contracts: A1 raster U-Net + GRU; A2 structured
  SE(2) U-Net/regional attention + GRU; A3 hierarchical graph message passing
  + regional attention + GRU. Do not let model outputs authorize illegal
  actions or alter AD* legality/safety semantics.
- Keep utility, nonnegative traversal penalty, behavior logits, legal
  categorical support, chosen sampled index and exact old log probability
  consistent through the command and per-tick execution trace. Route shaping
  and warm-start labels need their own source-bound data evidence; no P oracle
  or exhaustive legal-node coverage may be claimed without it.

**Acceptance:** per-track shape/order/mask/digest contracts, finite/nonfinite
failure behavior, sample/replay linkage, GPU residency, checkpoint identity
and representative route behavior pass. Compare A1/A2/A3 using identical
profile, input availability, seeds, D0/D1/D2 controls and resource budget.

### P4 — Plan C C1/C2 and candidate/route integrity

**Dependencies:** P1 schema, P2 candidate and route pipeline, accepted Plan C
and Route rows. The deterministic candidate generator and D1 comparator must
be the same for C1 and C2.

- Implement 537 stable source rows with raw/model widths 20/40, masks, ranks,
  ordinary K=16 selection, separate continuation mapping, and stable source
  row/command identity. Preserve oracle identity/sentinel domain separately
  from policy candidates.
- Retain 9,681 ordered route result rows, 152 direct-scatter chunks of at
  most 64, 73 path points and all capacity rows. Keep policy/oracle queries,
  geometry, status, feasibility, route variant, target handle, cost units and
  support digest bound exactly as reconciled. Candidate recall remains a
  diagnostic unless the frozen qualification protocol defines it as a gate.
- Implement C1 engineered candidate scorer and C2 raster map encoder plus
  candidate-to-map attention. Both predict gross strategic value; deterministic
  arbiter subtracts route/risk costs exactly once, enforces safety and applies
  fallback/hysteresis. No Plan A node penalty is introduced into Plan C.

**Acceptance:** canonical candidate ordering and source identity; candidate
recall/reference semantics fixed by protocol; no missing route rows or silent
truncation; same feasible candidates/features/route estimates for C1/C2/D1;
seeded simulator replay validates action execution and selected-versus-executed
records on each track; all compute remains on the two devices.

### P5 — PPO, end causes, horizons, RNG and atomic checkpoint continuation

**Dependencies:** P1 schema and artifacts, P2 device pipeline, P3/P4 interfaces,
and accepted P0a preregistration. Final source-bound protocol/artifact is P0b.
This package exclusively owns shared trainer edits to
`frc_defense/meco_track_training.py` to avoid conflicts.

1. Replace conflated `done=ended or timeout` with typed true-terminal,
   truncation, transition-finished and valid-bootstrap fields in the training,
   replay, counterfactual and evaluation call paths. Terminal takes precedence
   when both flags are set. Truncation bootstraps only from a valid final next
   observation; GAE recurrence never leaks across a terminal, truncation or
   reset. Execution failure has no bootstrap and may train critic only when
   all required transition data are valid.
2. Implement the reconciled fixed-horizon return target: 160 physics ticks,
   maximum 12 per macro and a permitted partial final macro, gamma 0.993 per
   macro with the adopted branch/continuation exponent convention. Bind
   end-cause and horizon semantics into dataset, checkpoint, training and
   evaluation identities. Regenerate incompatible counterfactual targets;
   do not relabel legacy outputs.
3. Implement two independent Philox4x32-10 streams using the exact reconciled
   uint16 stream and draw-site registries, deterministic draw ordering,
   rejected-word accounting, collision/overflow rejection, checkpointed
   counters and source-bound code registry. Preserve exact sampled action and
   old log probability, separate from semantic command and execution trace.
4. Make the complete rollout/update transaction atomic across model weights,
   optimizer slots/steps, PPO RNG counters and transition/artifact validity.
   Prevalidate/stage all rows and capacities or restore all state on any
   schema, digest, replay, RNG or capacity failure. Save one atomic checkpoint
   only at committed update boundaries. A physical simulator tick cannot be
   undone; after a post-tick failure, stop that worker and resume from the last
   committed checkpoint with frozen seed/environment reconstruction.
5. Write exactly five source/profile/protocol/device-bound checkpoint
   manifests. Resume rejects absent, malformed or incompatible manifests,
   optimizer state, RNG counters, profile, device identity or source; it never
   silently reseeds or resumes pre-change checkpoints.

**Acceptance:** golden transition/GAE fixtures cover terminal, truncation,
continuation, reset, execution failure and invalid next observation; digest
and RNG replay fixtures verify exact counters/bytes and cross-update resume;
fault injection at each validation/update/checkpoint stage proves all-or-none
software state; representative GPU simulator replay demonstrates the executed
result and end-cause record. No tests alone can close the behavior gate.

### P6 — five-lane training/evaluation, Stage 1 and dashboard evidence

**Source implementation dependencies:** accepted P0a and P1–P5. P6 owns the
qualification v5 schema, runner and source implementation (including
`frc_defense/meco_qualification.py` and its qualification tests) after P0a.
P6 source must be frozen and reviewed before P0b final calibration. **Runtime
qualification dependencies:** accepted P0b, P1–P6 integrated and reviewed,
fresh five-track checkpoints, frozen protocol v5, dual-device readiness and a
separate human authorization to start the new campaign. P0b does not require
P6 runtime results; P6 qualification cannot run before P0b ACCEPT.

- Start A1/A2/A3/C1/C2 from fresh source-bound runs under the same supervisor
  identity, assigned task/category/seed schedule, runtime, profile and device
  pair. Persist progress and checkpoint manifests per lane. One lane's failure
  has explicit sibling behavior; it cannot corrupt, silently restart, or
  advance another lane's checkpoint.
- Maintain uninterrupted training until the user manually stops it. After
  each committed update, publish immutable checkpoint snapshots on schedule.
  A continual read-only diagnostic evaluator can use those snapshots without
  resetting training state. Use declared GPU shares and never let evaluator
  writes alter the trainer's checkpoint or RNG.
- Only the frozen v5 qualification runner advances Stage 1. Each of the five
  tracks must satisfy every protocol-defined category, layout, held-out
  defender family, safety, paired D0 and D1/D2, score/progress, coverage and
  confidence-interval condition on the same checkpoint bundle and source.
  Run three consecutive passing rounds per track. Any failed case, protocol
  or checkpoint change resets the affected streak and requires a new immutable
  round report. Prior 100/100 diagnostics, partial updates, old-source runs,
  tests, and evaluation summaries do not count.
- Keep training and evaluation as distinct records. Freeze policy/checkpoint
  hashes before the final-test split is read. Final test cannot tune or select
  checkpoints. No metric is called official match score; the adopted outcome
  is simulated accepted active-HUB FUEL score.

**Acceptance:** all five tracks independently have three consecutive passing
Stage 1 rounds under the same compatible v5 contract and exact checkpoint
bundle, complete immutable reports and manifests, no source/device/protocol
drift, and continuous-run health records without unhandled error, hidden
fallback, corrupted checkpoint or unrecorded restart. Training itself remains
active until the user manually stops it; Stage 1 success does not auto-stop
workers.

## Shared commit, review, and integration cadence

- Planner pushes its complete plan and pins the commit/blob SHA before Pass 1.
- Source-only authoring begins only after package plan ACCEPT, applicable
  prerequisites (including P0a where required), exact isolated worktree/base,
  and named ownership. G0 gates all execution and integration actions.
  Make one coherent, reviewable commit per interface, kernel, error contract,
  or fixture; push each commit before beginning its dependent slice. Every
  commit records source/profile/protocol identity changes and migration impact.
- Pass 3 first records SOURCE/API ACCEPT or REVISE on the exact source-only
  candidate head. This candidate API pin cannot authorize dependent-package
  integration or runtime use. After G0, Pass 3 checks exact pushed
  hashes, allowlist, invariants, required tests, device traces and simulator/
  dashboard evidence; only this final ACCEPT allows integration. Persist both
  dispositions and do not accept a moving branch or stale commit.
- After ACCEPT, integration owner merges the exact reviewed head, records the
  integration commit and pins it as the next package base. Any conflict
  resolution creates a new head and must be reviewed again. Never push shared
  changes directly from the campaign checkout.
- Maintain an implementation ledger for package, owner, branch/worktree,
  base/head commit, paths, review-record hash, tests, simulator replay,
  dashboard freshness, GPU-residency evidence, checkpoint/protocol identity
  and remaining gates. Report unresolved gates plainly; never turn missing
  evidence into a pass.

## Required validation after implementation begins

The repository `AGENTS.md` requires representative end-to-end simulator
validation whenever tests check robot, scoring, collection, shooting, ferrying
or other game behavior. For every such package, pair tests with a seeded
simulator evaluation that exercises the changed behavior and inspect its fresh
simulation record/replay. Record scenario, seed, replay ID, source and result.

Before generating or reviewing any dashboard replay, restart serving dashboard
services after code changes; verify the backend and frontend are from the
current integration checkout, services are active and endpoints respond. Then
generate a fresh replay and verify provenance. Mark replay outputs made by old
code as outdated; do not use cached or pre-change replays as acceptance.
Report exactly which tests and simulator evaluations ran, replay/scenario IDs
and seeds, and whether dashboard backend/frontend freshness and endpoints
were checked. This plan has run none of those checks.

GPU-only validation must include full-path transfer/synchronization traces,
per-kernel device and memory evidence for both WX9100 units, explicit failure
tests for device loss and forbidden CPU fallback, update throughput and
evaluation interference accounting, and agreement with deterministic seeded
simulator records. Tests, compilation, profilers, dashboard restarts, simulator
evaluations and device runs require their own accepted package scope; none are
authorized in this planning tranche.

## Completion definition and explicit non-goals

The goal is complete only when the exact reconciled ABI/profile is implemented
without undocumented field/byte drift; the P0a preregistration is accepted
before source work and the P0b v5 Phase 0/calibration artifacts are
independently accepted and bound to the reviewed P1–P6 source; all five
source-bound checkpoints can resume with exact optimizer and RNG continuation;
the complete algorithmic path uses both approved WX9100 devices without CPU
fallback; representative simulator records and current-dashboard replays
validate each changed behavior; and A1, A2, A3, C1, C2 each pass three
consecutive Stage 1 rounds under the frozen protocol with complete artifacts.
Continuous training remains uninterrupted until the user's manual stop.

There is no arbitrary million-update completion target. Update count is a
training control, not the qualification criterion. D1/D2 remain matched
controls; optional deployment export/Hailo profiling does not substitute for
GPU-only training, simulator evidence or Stage 1. No source conformant status,
qualification, campaign health, or human approval is inferred from this plan.
