# MECO AutoDrive: Parallel Learning Architecture Evaluation

**Status:** Proposed experimental specification, not an implementation or validation claim
**Target:** FRC 2026 REBUILT, SystemCore with optional Hailo acceleration
**Repository:** https://github.com/MECO-Robotics/AutoDrive
**Companion:** `MECO_AutoDrive_Neural_Architecture_Plan.md` (original spatial neural architecture plan)

## 1. Decision and hypothesis

Continue **three Plan A spatial neural tactical-scoring architectures** and add **two Plan C candidate-level neural strategic-utility architectures**. Benchmark all five against identical deterministic baselines and common opponent scenarios. The intent is architectural diversity, not to assume that neural scoring outperforms explicit rules.

- **Plan A:** learn per-legal-pose goal utility `U(q)`, nonnegative traversal penalty `P(q)`, and five behavior logits (`drive`, `intake`, `score`, `ferry`, `defense`). AD* remains responsible for legal search and downstream execution. This is the original document's design.
- **Plan C:** deterministically enumerate feasible action/goal candidates, then learn their *strategic value*. AD* computes feasible routes and their costs. Plan C does not output per-node traversal penalties. Behavior is part of each candidate, rather than an unconstrained behavior decision detached from a feasible destination.
- **Common safety:** deterministic legal-space checking, footprint and swept-transition constraints, controller capture feasibility, dynamic collision handling, fallback pathing and interruptible execution. A neural model never authorizes an illegal motion or mechanism action.

**Principal research question:** Does full-field neural utility/cost shaping improve outcomes over learning only how to rank a small set of feasible, route-costed decisions?
**Secondary question:** Does either learned approach beat a deterministic strategy when perception, predictions, and available actions are matched?

## 2. Five models to evaluate

| ID | Track | Model | Primary learned outputs | Why include |
|---|---|---|---|---|
| A1 | Plan A | Multiscale 2D CNN U-Net + GRU | Dense legal-pose `U, P`; behavior logits | Deployment-oriented spatial baseline |
| A2 | Plan A | Structured SE(2) U-Net + regional attention + GRU | Dense legal-pose `U, P`; behavior logits | Tests topology-aware spatial representation |
| A3 | Plan A | Hierarchical SE(2) GNN + regional attention + GRU | Per-node `U, P`; behavior logits | Tests explicit graph propagation |
| C1 | Plan C | Candidate-feature MLP + GRU + shared candidate scorer | Utility / optional uncertainty per candidate | Small, easy-to-ablate strategic learning baseline |
| C2 | Plan C | CNN map encoder + candidate cross-attention + GRU | Utility / optional uncertainty per candidate | Tests whether field context helps candidate ranking |

A1–A3 correspond **exactly to candidates A, B, and C** in the companion document. Original candidates D (Graph U-Net) and E (GraphGPS-style) are deferred, not silently converted into Plan C. All parameter counts below are preliminary design targets, not measured implementations.

### A1: Multiscale CNN U-Net + GRU

Keep companion design: 2D six-channel dynamic map, seven global state scalars, temporal GRU(64), multiscale CNN encoder/decoder, 8-heading × 2 scoring channels, plus five behavior logits. Filter to legal poses. Nominal planning budget: **~70K–200K parameters**. Crucial limitation: static legal topology is not inherently supplied to CNN; a legal output mask alone does not make its latent representation connectivity-aware.

### A2: Structured SE(2) U-Net + regional MHA + GRU

Keep companion design: lift dynamic input to `(x,y,heading)` legal transition lattice, heading/transition-aware message passing and pooling, ~32–128 regional tokens, 4-head attention, GRU, decoder producing `U,P` on legal nodes. Nominal budget: **~100K–200K parameters**. Validate semantic consistency of heading wraparound, graph pooling, and whether fixed masked operators actually compile on target hardware.

### A3: Hierarchical SE(2) GNN + regional MHA + GRU

Keep companion design: fine legal graph message passing, connectivity-aware region pooling, attention over ~64 region tokens, GRU, light region-conditioned decoder and shared node heads. Nominal budget: **~105K–130K parameters**. Expect more difficult NPU deployment due to irregular gather/scatter and aggregation; CPU fallback is permitted only if measured latency is acceptable.

### C1: Candidate-feature MLP + GRU

**Purpose:** test whether learning context-dependent *decision values* is sufficient without learned spatial overlays.

**Inputs shared with Plan A:** identical observation timestamps, six raster channels and seven global scalars. A deterministic feature extractor may summarize the raster into candidate-relevant numeric features. If engineered features use privileged simulator information not available to Plan A, the comparison is invalid; track and disclose any additional feature. The prediction state and deterministic planner may consume measured velocity, but the *initial NN input* should honor the companion plan's no-explicit-NN-velocity ablation, or introduce the same motion channels for both tracks in a separately labelled experiment.

**Candidate set:** a deterministic generator emits a bounded set of action/goal tuples `(behavior, target pose/region, required constraints)`, such as collect a reachable FUEL cluster, score, ferry, traverse, position defensively, or continue an existing commitment. Candidate generation applies match-phase and mechanism legality, robot footprint, capture feasibility, estimated arrival/return time, and a safety mask. Include `continue current behavior` and a valid safe fallback where applicable. Require at least one valid option, else invoke deterministic fallback. Use a configurable top-`K` cap (initial experiment `K=16`, ablate `8/16/32`), sorted by deterministic candidate ID or stable canonical ordering, not a model-derived order. Candidate recall must be measured against a higher-budget oracle set; a model cannot select omitted options.

**Candidate feature vector (illustrative, ~16–32 scalars):** normalized goal pose and heading, behavior type embedding/one-hot, estimated AD* path travel time, pickup/scoring return time, expected FUEL opportunity, estimated defensive exposure/interception risk, goal crowding, current commitment/switching cost, confidence/age of detection, and phase legality flags. Do not mix oracle knowledge into deployed features. Record unavailable-feature masks. Critical quantities with uncertain estimates should be marked rather than silently set to zero.

**Network specification (first prototype):** global-state MLP (`7 → 64 → 64`), compact candidate MLP (`d → 64 → 64`), previous context GRU(64), concatenation or gated fusion, shared MLP head (`192 → 64 → 1`) across candidates; optional second head for uncertainty or quantiles. Parameter count to be computed from the fixed actual `d` and chosen modules. **Target scale:** tens of thousands to low hundreds of thousands of parameters. Use masked softmax or argmax over only valid candidates. Produce calibrated normalized utility, not a geometry override.

**Runtime contract:** AD* pre-computes feasible route cost for each candidate or uses verified cached route costs with age metadata. Score valid candidates with consistent value units, e.g. predicted incremental match points over a fixed horizon or expected scoring contribution net of travel/defense effects. Avoid subtracting route cost a second time when it was included in the learned target. Specify exactly one of: (a) model predicts *gross reward* and arbiter subtracts separately measured costs, or (b) model predicts *net value* and arbiter does not subtract them again. Start with **(a)** for auditability. Hysteresis and switching penalties remain deterministic. No learned traversal penalty.

**Primary ablation:** replace learned utility with a manually specified reward/route-time/risk equation using the exact same candidates and feature estimates. This is a strong baseline, not a strawman.

### C2: CNN map encoder + candidate cross-attention + GRU

**Purpose:** retain field-wide spatial understanding while producing only candidate utilities, rather than every SE(2) node's `U,P`.

**Inputs:** the same dynamic raster and global state as A1–A3, plus identical candidate table, masks and route-cost features as C1. Optional static legal geometry input must be disclosed and added to the closest A1 comparator in a controlled ablation. Preserve dynamic confidence/signed-time semantics and unobserved-state masks.

**Network specification (first prototype):** shallow 2D CNN encoder with strides or pooling to a compact map token grid; 64-dimensional map tokens; encode each candidate's features, goal pose and behavior into a 64-dimensional query; perform **candidate-to-map cross-attention** with 2–4 heads (not full attention over all ~10K legal poses); fuse seven global scalars and GRU(64) temporal context; shared 2-layer utility head per candidate. Start with `K=16` candidates, 32–128 spatial tokens and 64-dimensional width. Record exact token counts and end-to-end runtime. Prefer static-shape tensors and known NPU operators, but **do not assert Hailo compilation before testing**.

**Outputs and selection:** same contract as C1; masked per-candidate gross strategic utility and optional uncertainty. Downstream deterministic route feasibility, time/risk cost subtraction, safety gates and action arbiter are unchanged. This makes C1 vs C2 a reasonably isolated spatial-context ablation.

**Risk:** learned attention can still exploit shortcuts, map artifacts or spurious correlations; inspect attention only as a debugging aid, not as proof of causal reasoning.

## 3. Deterministic and predictive controls (essential)

Run the following non-neural comparisons before declaring either track beneficial:

- **D0: existing AutoDrive deterministic implementation**, recorded at a pinned commit, configuration and simulator seed list.
- **D1: matched deterministic candidate strategy**, with the *same* candidate generator, AD* route costs, eligibility constraints and estimated opponent risk used by C1/C2, but hand-specified weights and hysteresis.
- **D2: D1 plus a predictive opponent model**, first constant-velocity / multi-model kinematics. This isolates prediction gains from neural strategic gains.
- **D3 (optional): D1 plus learned opponent prediction**, held constant for both Plan A and C when explicitly testing opponent-prediction benefits. Do not give one candidate privileged motion forecasts while withholding them from its comparator.

Separate **prediction**, **strategic valuation**, **path planning**, and **local collision avoidance** in timing and logs. A trajectory predictor may be learned later; it is not silently bundled into either strategic track's reported gains. A spatial cost overlay is not equivalent to a time-indexed occupancy forecast.

## 4. Shared observation and planner contract

Retain from companion plan:

- Field raster **13 × 13 inches (0.3302 m)**, nominal ~51 × 25 grid, recompute exact size from configured field geometry.
- Six dynamic raster channels: `[fuel_confidence, fuel_signed_time, ally_confidence, ally_signed_time, opponent_confidence, opponent_signed_time]`.
- Seven robot/game scalars: `[x, y, sin(heading), cos(heading), estimated_fuel_load, signed_hub_timer, match_time_remaining]`.
- Temporal recurrence: GRU(64); reset hidden state on robot initialization, simulator episode reset and significant loss of tracking, with explicit stale-time policy.
- Full legal SE(2) graph available for deterministic constraints; global AD* search in current noted code is XY with swept SE(2) checks and local heading repair, **not already full global SE(2)**. Do not imply that A2/A3 models change planner search semantics automatically.
- Never loosen legal transitions, intake capture constraints, scorer heading control, opponent safety handling, mandatory passage orientations or cost-aware line-of-sight requirements.
- Asynchronous inference publishes versioned atomic snapshots including observation time, inference end time, model ID, candidate/pose ordering and validity window. Discard stale/mismatched predictions and continue deterministic behavior.

**Fairness warning:** Plan C gains engineered candidate features and potentially costly per-candidate AD* route estimates, whereas A1–A3 get dense spatial learning. Report total feature-extraction + inference + route-evaluation latency for each track, not NN inference alone. Hold fixed the available observation information and final action constraints; document residual differences rather than claiming an artificially identical input representation.

## 5. Training strategy

**Phase 0: freeze baseline and dataset protocol.** Pin robot dynamics, field geometry, legal connectivity, opponent tracks, update frequencies, sensor noise, match scoring, safety rules, and commit hashes. Partition held-out defender policies and maps/starting situations *before training*. Save independent validation seeds.

**Phase 1: implement interfaces without models.** A1–A3 use synthetic `U/P` fields and behavior scores; C1/C2 use scripted candidate ranking. Test masks, tie-breaking, stale snapshot fallback, cost units, route caching, arbitration and interruption.

**Phase 2: supervised warm start.** Gather deterministic expert trajectories and stronger simulation oracle comparisons where feasible. For Plan A, supervise legal-node goal/routing values and feasible behavior preferences. For Plan C, label candidate gross reward or pairwise preference using matched rollouts from candidate actions, accounting for simulation variability. Do not label an expert decision as universally optimal just because the deterministic baseline chose it.

**Phase 3: controlled RL fine-tuning.** Optimize executed rollout reward and risk penalties, not merely offline proxy accuracy. Ensure policy-gradient methods such as PPO see an actual stochastic distribution over tactical choices and their log probabilities; deterministic cost maps alone do not supply this. For A1–A3 specify a valid stochastic choice from feasible goals/behaviors or use an alternative optimization method. For C1/C2 a masked categorical policy over candidates is straightforward. Run matched environments, evaluation budgets and tuning resources; document differences in action-space expressivity.

**Phase 4: adversarial/generalization evaluation.** Train against stationary blockage, direct pursuit, trench interception, feints, reactive defenders, opponent-policy pools and multiple robot start states. Hold out entire defense policy families and not just random seeds. Test sensor delay and low confidence, abrupt opponent change, robot bumps and mislocalized initial pose.

**Phase 5: deployment profiling.** Export ONNX, test actual Hailo compiler/operator support and memory usage where possible, measure CPU fallback, transfer overhead, inference age, AD* route computations, action arbitration and effective score-to-actuation latency. Never declare any architecture deployable from parameter count alone.

## 6. Evaluation metrics and decision gates

| Category | Metrics |
|---|---|
| Game performance | Match points contributed; FUEL per second; scoring/ferry efficiency; failed collection attempts |
| Defensive robustness | Time lost to opponents; interception rate; collisions; successful alternate crossing; reaction to feints |
| Navigation | Total travel time/distance; feasible path rate; heading/footprint violations; replans; route oscillations |
| Strategic decisions | Regret vs simulation oracle on limited candidate sets; missed viable candidates; switching frequency; value calibration |
| Reliability | Performance after perception dropouts, model stalls, stale outputs and forced NN disablement |
| Compute | End-to-end p50/p95/p99 latency; worst-case replanning; memory; NPU/CPU transfer overhead; power if instrumented |
| Generalization | Held-out opponent families, start states, simulated teammate behaviors and environmental noise |

**Primary success criterion:** beat the strongest matched deterministic baseline on held-out match outcomes with confidence intervals across episodes and without increasing safety violations. Predeclare the minimum operationally useful improvement after measuring baseline variance; do not invent a universal percentage threshold. Use paired initial states/opponent seeds and report bootstrap confidence intervals. A model that only wins in training or on a handpicked sinusoidal defender does not advance.

**Secondary selection:** if C1 ties C2, favor C1 for reduced complexity. If Plan C matches or exceeds Plan A, prefer simpler candidate-level outputs unless Plan A exhibits a repeatable benefit specifically from learned *route shaping*. If Plan A wins, verify that the comparison is not due to incomplete candidate enumeration or extra information. Reject architectures whose latency or recovery behavior violates robot operating requirements even if offline reward improves.

## 7. A/B experiments that isolate the real question

1. **A1 vs C2:** similar raster CNN access, dense pose cost field vs candidate ranker. Hold map input constant as far as feasible.
2. **C1 vs C2:** quantify contribution of spatial map embeddings beyond engineered candidate features.
3. **A1 vs A2 vs A3:** quantify returns from explicit topology awareness and irregular graph operators.
4. **C1 vs D1:** does learned utility beat hand weights with the exact same feasible candidates and route estimates?
5. **D1 vs D2 vs D3:** do opponent forecasts help, and is a learned predictor better than kinematic/multi-model prediction?
6. **Matched prediction across all five models:** prevent mistaking better perception/prediction for a better tactical architecture.
7. **A-track penalty ablation:** run `P(q)=0` while retaining learned goal/behavior scoring to see if traversal cost shaping contributes independent value.
8. **Candidate recall ablation:** `K=8/16/32` and expanded oracle pool to diagnose Plan C's expressivity ceiling.
9. **Fallback test:** disable neural inference mid-run; both tracks revert to the identical dynamic-aware deterministic strategy.

## 8. Implementation boundaries and next actions

Relevant existing entry points, as identified in the companion plan, are `frc_defense/adstar.py`, `frc_defense/tensor_adstar.py`, `frc_defense/_tensor_3v3_actions.py`, `frc_defense/_tensor_3v3_navigation.py`, and `frc_defense/gamepiece_actions.py`. These are **references from the source plan, not a fresh code audit**.

Recommended order:

1. Freeze baseline and simulation fixtures and write the common decision/observation schema.
2. Implement Plan C's deterministic candidate enumerator and D1 benchmark first, since it is needed for fair neural-ranker evaluation regardless.
3. Implement a generic model adapter supporting both `dense_pose_scores` (Plan A) and `candidate_scores` (Plan C), with timestamps, schema validation and deterministic fallback.
4. Prototype **A1 + C1 first**, then **C2 + A2**, then A3 if earlier topology tests warrant it. Maintain all five as planned experiments, not mandatory production dependencies.
5. Train, compare and promote only based on the decision gates in Section 6.

## 9. Unresolved design decisions to measure rather than guess

- Exact legal graph dimensions and whether 8 heading bins preserve narrow-passage legality.
- Candidate generator coverage, optimal `K`, and frequency/cost of re-evaluating per-candidate AD* routes.
- Whether traversal penalty overlay needs time-conditioned edge costs to use moving-opponent predictions coherently.
- Whether learned gross reward should be predicted over a fixed time horizon or until next tactical decision; enforce consistent discounting.
- Availability and calibration of dynamic opponent state estimates, including tracking identity through occlusion.
- Whether A2/A3 NPU operators are supported on the actual Hailo device and whether their CPU timings meet the latency budget.
- How to constrain learning targets so a model never benefits in simulation by exploiting unrealistic detector or opponent behavior.

**Bottom line:** keep the three original, diverse Plan A models while testing two materially simpler Plan C approaches. The decisive comparison is end-to-end scoring and defensive robustness against well-matched deterministic baselines, not neural parameter count or offline prediction loss.
