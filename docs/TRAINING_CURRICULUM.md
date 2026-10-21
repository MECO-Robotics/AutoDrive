# Training curriculum: collection, phase-aware delivery, and defensive pressure

Date: 2026-10-20

Status: curriculum infrastructure implemented; scenario budgets, thresholds, scripted references, and learned policies remain unqualified.

## Objective and priorities

Train a robot that reliably collects FUEL and converts it into useful team outcomes under the current HUB state:

- **Friendly HUB active:** collect, carry between zones as needed, return to a scoring position, and shoot.
- **Friendly HUB inactive:** collect eligible FUEL outside the friendly alliance zone and ferry it into that zone for later scoring.
- **HUB state changes:** reassess the current load, location, remaining time, and route before continuing the previous objective.

Here, **zone transfer** means the robot physically carries FUEL between zones. A **ferry transfer** releases FUEL toward the friendly alliance zone. These are separate skills with different success conditions. HUB activity always refers to the learner's friendly HUB.

First establish undefended competence, then add defensive pressure. Learning to play defense is a later, separate priority. Full matches become a training environment after the component skills and phase transitions pass their gates.

The active/inactive split defines the initial teaching tasks. Later mixed scenarios should permit useful exceptions when the action is legal and produces a better team outcome. Do not encode a permanent rule that ferrying is always wrong during an active period or that carrying is always wrong during an inactive period.

## Review of the current setup

| Current capability or limitation | Implication for this curriculum |
|---|---|
| Production training entry points restrict learned policies to defense; deterministic offense supplies the opponents. | Supporting learned offense in the normal training path is prerequisite work. |
| The existing two-robot PPO curriculum advances through opponent pools according to training progress. | Add outcome-based promotion for the proposed offense curriculum; elapsed updates alone do not demonstrate competence. |
| Architecture scenarios include simple pickup, scattered pickup, focused defense, 3v3 defense, and experimental offense against a defender. | Reuse pickup setup and reset machinery, but add the missing transport, transfer, undefended cycle, and phase-transition tasks. |
| Experimental neural offense masks ferrying and interception. | It cannot currently qualify inactive-HUB ferry behavior. Add a dedicated safe ferry task and reward accounting before enabling ferrying in mixed tasks. |
| Some execution refinements are gated on deterministic offense. | Establish shared execution semantics or explicitly qualify differences before using teacher throughput as a promotion threshold. |
| The architecture study improved collection with little sustained scoring; its short training probes ended before the first HUB transition. | Teach delivery directly and evaluate phase transitions explicitly. Extend episode horizons to fit complete cycles. |
| Strategic policies have eight actions and four collection candidates. | Confirm each scenario's required target and alternate route are expressible through the available actions. A visible cluster alone does not guarantee the learner can select it. |

Sources: [training runtime](../frc_defense/tensor_training_runtime.py), [PPO curriculum selection](../frc_defense/tensor_training_ppo.py), [3v3 training](../frc_defense/tensor_training_3v3.py), [architecture scenarios](../frc_defense/architecture_scenarios.py), [experimental offense](../frc_defense/architecture_offense_scenario.py), and [architecture findings](NN_ARCHITECTURE_FINDINGS.md).

These are source and recorded-study observations. This document does not claim that the proposed curriculum has been implemented or passed simulator evaluation.

## Stage progression

The scenario names below are proposed identifiers, except for the existing `pickup_simple` and `pickup_scattered`. Every scenario needs a versioned definition, reproducible seed, success criterion, failure criterion, and time budget.

| Stage | Scenario family | Exercise | Completion evidence |
|---|---|---|---|
| 1. Simple pickup | `pickup_simple` | One nearby ball, then a small cluster; vary starting heading and distance. | Required distinct FUEL acquired and retained within the time budget. |
| 2. Scattered pickup | `pickup_scattered` | Scatter within one zone; vary density, distribution, and initial visibility. | Required fraction of available FUEL collected without repeated release/reacquisition credit. |
| 3. Multi-zone pickup | `pickup_multi_zone` | Place separated clusters in multiple legal collection zones; include empty and depleted zones. | Find productive zones, leave depleted ones, and collect the required amount. |
| 4A. Active-HUB transport | `active_carry_to_zone` | Start loaded outside the scoring zone. Qualify trench and bump crossings separately, then route choice. The real alliance boundary has no independent open crossing; open loaded approaches belong to stage 4B. | Reach a valid scoring area with the required load retained before the active window closes. |
| 4B. Active-HUB shooting | `active_shoot_preloaded` | Start loaded near a valid scoring position; increase approach distance and heading error. | Required actual HUB scores during the active window. Arrival or a shoot request alone is insufficient. |
| 4C. Inactive-HUB ferry | `inactive_ferry_preloaded` | Start loaded at a valid ferry position; add approach, alignment, different loads, and landing conditions. | Required distinct FUEL lands in the intended friendly staging area and remains available at a defined settling interval. |
| 5A. Active-HUB cycle | `active_collect_carry_score` | Collect, carry between zones as needed, shoot, and return for another batch. | Complete the required cycles and achieve useful scored throughput. |
| 5B. Inactive-HUB cycle | `inactive_collect_ferry` | Collect eligible FUEL outside the friendly alliance zone, ferry home, and repeat. | Complete the required cycles with net useful staged FUEL and no recycling exploit. |
| 6. Phase transitions | `hub_transition_cycles` | Begin before or during active/inactive boundaries, with varying load and location. | Switch objectives appropriately and convert staged FUEL into scoring after activation. |
| 7. Defensive pressure | `offense_vs_scripted_pressure` | Repeat qualified cycles against increasingly capable blockers. | Retain useful throughput, choose feasible routes, and recover from interruption. |
| 8. Team integration | `team_delivery_cycles` | Add a scripted receiver/scorer, then more teammates and opponents. | Deliveries become team scoring; robots share routes and collection areas productively. |
| 9. Full matches | `full_match_offense` | Use the real match schedule with scripted teams, then broaden opponents and starting conditions. | Sustain team contribution across phases and team compositions on unseen scenarios. |

Stages 4A, 4B, and 4C are complementary branches. Introduce short preloaded shooting and ferry exercises once stage 1 works, alongside stages 2 and 3. This supplies successful delivery experience early while collection difficulty increases. Joining each branch into stage 5 requires its relevant prerequisites to pass.

## Scenario construction

### Teach the end of the cycle early

For shooting, work backward from a successful finish: preloaded at a scoring position, approach and shoot, carry from another zone and shoot, then collect and complete the cycle. For ferrying, use preloaded release, approach and release, then collect and ferry.

Choose task time budgets from measured successful scripted runs with margin. A complete-cycle episode must leave time to collect, traverse, deliver, and demonstrate repetition. Also retain a firm timeout so waiting forever cannot pass.

Keep realistic physics, sensing, capacity, and robot capabilities. Start with simple layouts and narrow initial-state variation; broaden geometry and uncertainty after the basic task succeeds. Policy observations must use simulated perception and permitted telemetry. Ground truth may initialize scenarios and calculate outcomes, but must not populate learner tracks or teacher inputs with hidden information.

### Make HUB state an explicit training variable

Isolated active and inactive exercises can use controlled phase windows, provided the scenario records the override and uses the normal state-update path. Later stages use the actual match schedule. Do not set a flag once if the match clock will overwrite it on the next step.

Confirm the policy can observe friendly HUB activity and any phase timing made available to the deployed policy. If time until transition is unavailable, do not require behavior that depends on knowing it exactly.

Transition scenarios must include:

- Active to inactive while collecting, while carrying home, and while approaching a shot.
- Inactive to active before ferry release, after release, and with useful FUEL already staged at home.
- Full, partial, and empty loads on both sides of each boundary.
- Enough remaining active time to score, and too little time to complete the same route.
- Empty collection areas and situations where holding a useful load is preferable to a pointless transfer.

Record objective changes and eventual outcomes. Avoid rewarding immediate action switching by itself; finishing a nearly complete useful action may be the correct choice. Judge the resulting delivery, scoring, and wasted travel.

### Add defense one difficulty at a time

| Pressure level | Defender behavior | Learner skill |
|---|---|---|
| 0 | No defender. | Establish the scenario's achievable throughput. |
| 1 | Stationary obstruction with a feasible alternate route. | Route around a blockage. |
| 2 | Fixed back-and-forth patrol with seeded initial phase. | Time passage, wait briefly, or reroute. |
| 3 | Trench blocker, then bump blocker, then seeded choice of blocked route. | Recognize route-specific delay and use an available alternative. |
| 4 | Deterministic reactive defender that responds to observed learner motion. | Recover from interception and change route or collection target. |
| 5 | Mixed scripted styles, speeds, positions, and multiple opponents. | Generalize beyond a particular patrol or defender. |

Apply pressure to both active scoring and inactive ferry cycles. First qualify each route without opposition. If a layout has no feasible passage, label it as a recovery or waiting task with a suitable outcome rather than treating failed traversal as a learning failure.

Use seeded setup and scripted-policy versions for reproducibility. A deterministic defender may react dynamically to the learner; it does not need a fixed trajectory. Reserve unseen patrol phases and configurations for evaluation.

## Outcomes and reward accounting

Promotion uses physical outcomes, not accumulated training reward or requested action counts.

| Skill | Primary outcome | Useful diagnostics |
|---|---|---|
| Pickup | Distinct required FUEL retained. | Time to first pickup, fraction collected, idle time. |
| Loaded transport | Required load delivered to the destination zone. | Load loss, travel time, route, stalls. |
| Shooting | Actual FUEL scored while the friendly HUB is active. | Time to first score, scored fraction of load, wasted attempts. |
| Ferrying | Net useful FUEL delivered from outside the friendly alliance zone into the intended staging area. | Landing distribution, retained staged quantity, immediate losses or recollection. |
| Complete cycles | Scored FUEL per active second or useful staged FUEL per inactive second. | Completed cycles, cycle duration, unsuccessful deliveries. |
| Transitions and team play | Team scoring and useful stock remaining, compared on paired scenarios. | Time from activation to score, wasted travel, duplicate work, congestion. |

Track physical piece identity plus its delivery/scoring history. A ferry can earn delivery credit once for a genuine outside-to-home movement. Recollecting and releasing the same staged piece must not create another credited delivery. A genuinely scored piece that later returns to play starts a new eligible lifecycle; do not permanently disqualify it from legitimate later cycles.

Use bounded intermediate rewards for acquiring, retaining, and making progress toward a useful delivery. Limit their total contribution so collecting or moving indefinitely cannot beat task completion. Scoring and ferry outcomes remain distinct. A solo inactive-HUB task needs direct delivery credit because eventual teammate scoring may lie outside its horizon.

Initially mask actions unrelated to a primitive. In mixed tasks, restore relevant legal choices and evaluate whether they help. Always preserve legal-action masks; record masks alongside decisions so forced behavior cannot be mistaken for learned choice.

## Promotion, retention, and evaluation

These are proposed starting thresholds. Qualify the scripted reference and freeze each scenario's thresholds and time budget before judging learner checkpoints.

- Primitive tasks: at least 90% success in each required scenario category across three consecutive validation evaluations.
- Complete cycles: at least 90% success on the configured cycle objective and at least 80% of qualified scripted-reference useful throughput on paired seeds.
- Pressure tasks: at least 80% of the qualified reference's useful throughput under the same pressure, plus a scenario-specific absolute delivery minimum. A zero-performing reference cannot establish readiness.
- Retention: earlier prerequisites must continue to meet their success thresholds; investigate a useful-throughput drop greater than 10% from the last qualified checkpoint.
- Report active/inactive, trench/bump, and each blocker category separately. An overall mean cannot compensate for a failed required category.

Start with at least 100 validation episodes per required category, report success counts and uncertainty, and expand the set when results sit near a threshold. Repeated validation is used for promotion and is therefore tuning data. Keep a separate final test set untouched until the curriculum/checkpoint selection is frozen.

Begin training mixtures around 70% current tasks, 20% qualified earlier tasks, and 10% next-stage practice where prerequisites are available. Redistribute the next-stage share if it produces no useful learning events. After promotion, retain earlier active and inactive tasks. Save the last qualified checkpoint; a failed stage triggers more prerequisite practice rather than automatic advancement.

Validate with both familiar and unseen layout families, not only new random jitter. Compare learner and reference on identical seeds, physics, sensing, robot capabilities, action cadence, time windows, and budgets. Evaluate independent training seeds and report per-seed results; many worlds from one trained policy do not replace independent training runs.

## Deterministic baseline and teacher

Use the deterministic algorithm as a baseline first. Qualify successful examples for every primitive and cycle, including phase transitions and both transfer modes.

For imitation, record the learner-visible observation, legal-action mask, selected strategic action, task identity, HUB state, and outcome. The teacher must use the same available information and executable action interface as the learner. A privileged diagnostic controller may identify achievable outcomes, but must be labeled separately and cannot silently provide imitation labels.

Suggested sequence:

1. Behavior-clone qualified demonstrations to establish successful action sequences.
2. Run the learner and ask the teacher for labels in states the learner actually visits, especially recovery states.
3. Continue PPO with a decaying imitation contribution while measuring actual task outcomes.
4. Evaluate the learner alone, with teacher interventions disabled.

Do not require exact action agreement when several choices work. Measure completed cycles and useful throughput. Shared execution must be qualified before a performance gap is attributed to policy quality.

## Full matches and later defensive learning

Enter full-match training after both active/inactive cycles, phase transitions, and the required pressure categories qualify. Keep primitive and cycle tasks in the training mixture. Initially use scripted teammates and opposition so failures can be traced, then vary team composition, defender styles, initial state, and supported robot capabilities.

Full-match evaluation must cover the complete schedule, actual encounters, staged-FUEL conversion, and multiple seeds. Match points are primary, with phase-specific collection, delivery, and scoring diagnostics. A policy that never encounters a defender has not qualified against that defender.

Learning to defend is a later branch: obstruct a predictable loaded attacker, contest a route, react to changing approaches, then coordinate with teammates. Use separate role-conditioned evaluation and checkpoints as appropriate. Once both roles are competent, add role allocation and mixed-role full matches.

## Implementation order

1. Add a supported learned-offense training path and qualify shared action execution, observation availability, and scenario resets.
2. Add lifecycle-aware delivery metrics and versioned scenario definitions, budgets, and seed partitions.
3. Qualify preloaded shooting, loaded zone transport, and inactive-HUB ferry tasks alongside existing pickup tasks.
4. Add multi-zone pickup, complete active/inactive cycles, and outcome-based promotion with prerequisite retention.
5. Add phase transitions, scripted pressure, and deterministic teacher datasets.
6. Add team integration and full-match training. Keep learned defense as the later branch.

Keep the architecture fixed during the first curriculum qualification. Compare architectures again after the tasks provide frequent delivery events and genuine encounters, using matched training budgets and identical promotion rules.

## Simulator validation and review record

Every behavior test run must be accompanied by a representative end-to-end simulator evaluation that exercises the changed behavior. Inspect the simulation record or replay for actual outcomes. Unit and regression tests alone do not qualify a stage.

Before generating or reviewing a dashboard replay, restart the relevant serving services after code changes. Verify services are active, their endpoints respond successfully, and served frontend assets match the current checkout. Generate a fresh replay, verify source/checkpoint provenance, and mark old-code records outdated.

Retain this evidence for each qualification:

- Curriculum/scenario version, checkpoint, source identity including working-tree content, physics settings, and action cadence.
- Scenario/replay identifier, seed, learner/reference configuration, time budget, and HUB schedule or override.
- Tests run, end-to-end outcome, gate metrics, and inspected event or trajectory evidence.
- Dashboard backend/frontend freshness checks, or an explicit statement that these were not completed.
- Promotion decision, category failures, and known limitations.

### Draft review, 2026-10-20

| Review concern | Resolution in this draft |
|---|---|
| Active transport and inactive ferrying were too loosely grouped. | Separate branches and outcome definitions, joined by explicit transition training. |
| Pickup could improve without useful scoring. | Preloaded finish tasks, backward task expansion, and complete-cycle gates. |
| Ferrying could reintroduce reward recycling. | Net outside-to-home delivery accounting with physical identity and lifecycle rules. |
| Fixed training duration could promote an unready policy. | Outcome-based promotion, separate category gates, and prerequisite retention. |
| Teacher advantages could be mistaken for student failure. | Require comparable observations, actions, execution, and paired evaluation. |
| Scripted defense could be memorized or physically impossible. | Held-out configurations, separate route qualification, and explicit recovery tasks. |
| Short opening tasks could be presented as full-match competence. | Separate transition, team, and full-schedule match qualification. |

Review performed through source/document inspection and consistency checking. No behavior tests, new simulator evaluations, replay generation, or dashboard freshness checks were run for this documentation-only change. All proposed thresholds and scenario families remain unqualified until implemented and evaluated.


### Implemented infrastructure, 2026-10-20

The versioned registry and seed partitions are implemented in `training_curriculum.py`; native scenario setup and lifecycle outcomes are in `curriculum_scenarios.py` and `curriculum_metrics.py`. `curriculum_training.py` adds standard ActorCritic PPO, independent paired evaluations, checkpoint/resume, and observed-teacher JSONL export. The [README commands](../README.md#opt-in-offense-curriculum) cover planning, a short update/resume, evaluation, and export. Both `frc-defense curriculum ...` and `python -m frc_defense.curriculum_training ...` use the same runner.

Normal-budget collection finishes a physical episode or reaches its scenario horizon before the update-boundary reset; `rollout_steps` is a minimum there. A supplied horizon is diagnostic and cannot promote. Horizon truncation bootstraps the critic from the final pre-reset observation; physical completion does not. Frozen worlds are excluded from subsequent PPO/value/entropy/imitation rows. Rewards are consumed on every 20 ms physics tick, including held strategic actions. Whole-batch reset uses `reset_curriculum_scenario`, without legacy `reset_done`.

Resume stores policy and optimizer state, CPU/CUDA and Python RNG, scheduler RNG, update/seed cursors, current/recovery tasks, per-category curriculum baselines and protocols, and source/runtime fingerprints. Atomic update-boundary checkpoints avoid a simulator-state restore requirement. Source identity includes Python and native simulator code plus runtime overrides. Only output location and target update count may change on resume. Every promotion retains an immutable `qualified/<scenario>/<policy_id>.pt`; historical qualified states refer to those artifacts. Failed prerequisite retention schedules recovery practice without erasing the saved baseline.

Validation runs use identical independent episode seeds, physics, sensor/controller interface, capacity, cadence, and horizon for learner and teacher. Each category reports success counts, a 95% Wilson interval, and measured useful throughput. A reference is qualified only after at least 100 successful-enough normal-budget episodes with positive useful throughput. Any horizon override, incomplete category selection, or fewer than 100 episodes is diagnostic and cannot promote. Full promotion still needs all required categories and three compatible consecutive rounds. Final test execution requires a selected checkpoint plus an explicit final-test flag; its exact bytes are frozen before reading test worlds.

The teacher uses sensed candidates and own telemetry, with generic NN action execution. It is not automatically a qualified reference. Pickup capacity is 30 and staged delivery capacity is six; actual capacities, robot types, legal actions, controlled schedules, and pressure setup are recorded with each episode. No deployed 60-FUEL policy or complete 160-second match is qualified. Reported useful throughput is per elapsed scenario second; phase-specific elapsed seconds are separate diagnostics. All draft targets and budgets still require reference qualification.

Current-source infrastructure records include [installed CLI, CPU PPO/update/resume, paired evaluations, and teacher export](../evaluations/curriculum-infrastructure-20261020/root-cli-final-validation/summary.json), [the actual HUB boundary, seed 17](../evaluations/curriculum-infrastructure-20261020/phase-boundary-final-seed17.json), and [CUDA1 training, resume, and physical transport](../evaluations/curriculum-infrastructure-20261020/root-cuda1-fixed-validation/summary.json). Earlier records whose source fingerprints differ from the checkout are marked outdated. These are infrastructure checks, not stage qualification. No dashboard replay was generated, and dashboard services/frontend freshness were not checked.

The [final validation manifest](../evaluations/curriculum-infrastructure-20261020/FINAL_VALIDATION.json) consolidates 79 focused checks across three runs, the current-source simulator evidence, test-seed reservations, retained failures, and qualification limits. Its [final CPU CLI summary](../evaluations/curriculum-infrastructure-20261020/root-cli-final-validation/summary.json) records the installed entry point and complete short update/resume/evaluate/export flow.

The latest accelerator checks use the explicit `cuda:1` device: [training and resume](../evaluations/curriculum-infrastructure-20261020/root-cuda1-fixed-validation/summary.json) both changed policy weights, and [loaded trench transport](../evaluations/curriculum-infrastructure-20261020/root-cuda1-fixed-validation/carry.json) retained six FUEL with an observed route crossing in 40 physics ticks (0.8 seconds), using validation seed `6360399568852484097`, master seed `1702`. Three device-selection ordering tests passed after the fix. The original [initialization failure log](../evaluations/curriculum-infrastructure-20261020/root-cuda1-validation/cuda1-train.stderr.log) is retained: native execution used GPU 0 while tensors were requested on GPU 1. The factory now selects the explicit CUDA device before simulator allocation. Earlier records with different source fingerprints remain historical; these checks do not qualify a curriculum stage or a complete match.

The final CPU sweep completed a two-world PPO update and resume, paired learner evaluation, and teacher export. Its shooting reference scored four FUEL on validation seed `2925952205389299713`; the inactive ferry reference retained four useful staged FUEL on seed `6712735917758480385` (master seed `1701`). The separate phase scenario began at an explicitly recorded initial match time of `3.9` seconds, crossed the ordinary `4.0`-second activation boundary, and scored four FUEL afterward, succeeding after 85 physics ticks. Recorded test runs passed 65 native/curriculum regression tests, 11 runner tests, and three device-order tests. The installed `frc-defense` entry point initially resolved an older checkout; refreshing the current virtual environment's editable install, without dependency or runtime source changes, made the installed command pass the same final sweep.

Test-seed hygiene: an early infrastructure freeze test exercised `pickup_simple/single`, master seed `0`, namespace test index `0` (seed `7069911598172733442`), for one physics tick. It did not train or promote a policy. Final-test evaluation now reserves index `0` of every test namespace for infrastructure diagnostics and starts at index `1`; the raw seed-partition API is unchanged. Freeze unit tests mock episode execution. No further real test-split smoke is run.
