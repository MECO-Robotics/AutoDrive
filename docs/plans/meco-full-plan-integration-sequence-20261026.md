# Pass 1 plan: integrate the five-track learning plan in reviewed dependency order

**Role:** Pass 1 planner artifact; integration sequencing only. This plan does
not implement, merge, rebase, test, import, execute, train, evaluate, restart a
service, touch a campaign/checkpoint, or access a GPU/simulator.

**Source-integration baseline:** `160911abe1cf4644628053dae054e363b7fb010b`.
The sequence was first integrated as a plan-only change at
`3861b56b24d9e88dedd4b6fbf3458162b72177ca`; that is an earlier integration
point, not current `origin/main`. Current `main` and `origin/main` are both
`61d3731e1531825ee7524065a158ba4765da0a84`, checked from the local branch,
remote-tracking ref, and remote branch. This head includes the accepted Plan A
source integration at `daa2663fc2fecdb183c304bc4dff41bcba9a1acb`, the accepted
Stage 1 preflight integration at `bfeb1294c99f3d656a9f53f20077cbdb8dcdc7cc`,
and the Plan C K-prefix repair at its exact reviewed candidate SHA. Retain
`160911abe` as the source-integration baseline; neither the earlier `3861b56`
document integration nor the current planning head changes that baseline. The
original main worktree had one existing user-owned modified file,
`frc_defense/csrc/tensor_collision_multi_hip_kernel.hip`; it is outside this
plan and must remain untouched.

**Normative source:**
`docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md`, SHA-256
`718ef9d76f5525bf8e4759d3a0a46790a3c978e2b146efec6436c89dbbb49b10`.
Current execution ledger at the planning base:
`docs/MECO_FULL_PLAN_EXECUTION_STATUS.md`, SHA-256
`469ff959001ff1947b2e4dc03699341aa232f94d5cc607a82d43f47532e07abf`.
Whole-compute GPU-only boundary:
`docs/plans/whole-compute-gpu-only-meco-plan-20261025.md`, SHA-256
`0fc594792e233c471d3e81c871e595b09bb13f039ddc7f6996afbda389cbaab0`.
These files, their section 1–9 requirements, the current curriculum version,
and the exact source/candidate heads below are the integration inputs. Re-read
them at Pass 2 against the then-current `main`; any changed bytes, moved source
head, or new candidate invalidates this sequence until Pass 1 amendment and
Pass 3 approval.

Additional pinned source-spec documents at the planning base are
`docs/MECO_QUALIFICATION.md` (SHA
`47a9ed0753df0dd456b4965f1dc8fe3e4c84961f0c5835e7e5c39869655bda1f`),
`docs/MECO_EVALUATION_PROTOCOL_BINDING.md` (SHA
`f368963f59f9ae8eb978bf2aca518a666ab7374f3bc4abb4884e3ce7aef3730e`),
`docs/MECO_ABLATION_MATRIX.md` (SHA
`4fe8e0e92472208d12587d346ee0bcd367ee5c7ed25dcd7510947da05f12dc0a`),
`docs/MECO_SUPERVISED_WARMSTART.md` (SHA
`f37a28711e0f4213ccb5787db7609d2274d2387970644dc3b64976cc9dd6b9c5`),
`docs/MECO_PHASE4_FACTORS.md` (SHA
`47d5c5c8050550b81a596c3a866834b492a9a33f3fd92a811e990d94b23f1bb4`),
`docs/MECO_PHASE4_RUNTIME_STATUS.md` (SHA
`79f3771ee58cbfbfaa689d736eadc695592f341e53c60153b7e9e90cab542e9c`),
`docs/MECO_TRACKS_IMPLEMENTATION_STATUS.md` (SHA
`5aff7f4bb356fb1f0beb37b33e46d6ae6c54815e6e327b8826bf67e8ffcfa372`), and
`docs/MECO_THREE_PASS_PROVENANCE.json` (SHA
`37e37b14b7609f303823d83c2240c9c14ebdbe848eff6621964f8cb405ca16b`).
These records are evidence pointers, not substitutes for exact source,
independent review, device, simulator or qualification evidence.

**This planning patch allowlist is exactly one path:**
`docs/plans/meco-full-plan-integration-sequence-20261026.md`. This Pass 1
authorization does not permit editing code, live campaign inputs/outputs,
checkpoints, dashboard sources, execution status or the provenance ledger.

## Objective and completion boundary

Integrate the already planned and partially implemented work needed to run and
compare **all five** approaches named in the source plan: A1, A2, A3, C1, and
C2. Preserve the user's strict “offload everything to GPU: no CPU” requirement
as an end-to-end compute boundary. Complete the source/protocol/evaluation
gates needed for all five to pass the first curriculum stage and then continue
training without qualification or operational errors. This integration plan
does not redefine that objective as “all code merged” or “one successful
checkpoint.” Qualification, architecture wins, promotion, and whole-compute
GPU-only claims require their own evidence and must not be inferred from Stage
1 completion.

## Pass 1 sequence refresh (2026-10-26)

This amendment supersedes the candidate state and dependency ordering in the
snapshot inventory and numbered integration order below wherever they conflict
with the exact heads and states in this section. The older rows remain historical
context; they are not authorization to merge, execute, or treat a former
candidate as current. It was prepared from planning base
`5afe639760a3e5ebef4ee392901dca55aa5e15c8`; the superseded document at that
base had SHA-256
`d7b7bd92041f9d9c09155ced6f8b6852a8584a0c413a4a1dd0aa7f35340a4ae0`.
Record the amended document's resulting SHA-256 in the exact-head review result;
the document cannot contain its own final content hash without a self-reference.

Every listed commit was confirmed in the local object database and its full ID
was checked. Plan artifact hashes below were computed from the artifact bytes
at the pinned commit. An accepted plan is planning authority only: it does not
mean its source exists, is complete, is eligible for integration, or is safe to
run. Any implementation integration requires a fresh independent Pass 3 on the
exact candidate head against the then-current base and approved plan. A prior
partial acceptance does not authorize a later or broader tree.

| Workstream | Current Pass 1 authority (full commit; artifact SHA-256) | Current source candidate | Scope and state |
|---|---|---|---|
| Shared command/transition ABI | Accepted field-table baseline `1c7e9ef02030fa519325c44aa12d4b07e5d171cc`; `docs/plans/gpu-shared-command-ppo-transition-abi-20261026.md`; SHA-256 `65a4d936deb792b26b61b7569fec992dc8d35ec9da11c02845fcf342fa4b7f30` (Pass 1 + Pass 3 accepted as planning artifact). New route/Plan A status amendment `6d46846b87a7790396ef5e45868869c66ac173f6`; same document; SHA-256 `19349ebd40bbe6c0f11e8001bb716524d0273f8f91862f040096883faf3eb338` (Pass 1 and Pass 3 accepted as a planning amendment) | Contract/spec only; `6d46846` is not an ancestor of current `origin/main` | The accepted baseline remains `1c7e9ef`. The Pass 1/Pass 3 accepted amendment specifies route result codes `0..3` and distinct Plan A HIP status codes `0..4`. It leaves route-required action classes, no-route encoding, and HIP status granularity unresolved. This plan-text acceptance does not authorize implementation: matching owner decisions and the unresolved V3 capacities, lifecycle, RNG, digest, and opaque-record fields still block schema shell/producer/consumer implementation. The original ABI base `947fc80ebe046363b64583adcc4c17da186309b5` is historical context. |
| Plan A device decision/support | Current accepted tensor-root amendment `c0e9ae958108af3c56c4b64a709d9a4237cc7a3c`; `docs/plans/gpu-resident-plan-a-decision-slice-20261025.md`; SHA-256 `01e2507a1d4d458a2c8348323e119395d6f4953d988a53c4a7df2395c06f1c8a`. It extends prior accepted amendment `9227d66b75cb98b5b0ce329a6e625263b812e0c1` (SHA `b7a718cde176d8b5eeca62f277580d0cbf200468803373089573aec8e68c69fb`). | `15f7c5b344191b886d4bd0d375d01e4810080dd7` | Plan text is Pass 1 + Pass 3 accepted for scoped source prep. Code has bounded partial Pass 3 acceptance only as static source prep; rejected as complete support/status integration. HIP return status is discarded and the support path still raises; canonical command `decision_valid`/`route_status` is not wired to simulator consumers or PPO V3. Not complete A1/A2/A3 or dispatchable. |
| Plan C K-capped model view | Normative amendment `8556c82b7715499b935bc25f067f2a5d0dfac75e`; `docs/plans/gpu-plan-c-k-capped-model-view-contract-20261026-amendment.md`; SHA-256 `782279be4857e849f9f90d2f6e938b22b244d34b5567f562847e61b88ebf4fb9`; accepted review-status refresh `4cdc30f923c0d948b43c57d452c4ee4b17cf61d9` / SHA-256 `6d9c0e6320acd503c8595fa619e8beeba24af88f36e49337d7325331a899ff0b` | Partial model-prefix source support in `61d3731e1531825ee7524065a158ba4765da0a84`; no standalone implementation head for this two-view contract pinned | Plan text accepted. The Plan C route-decision candidate adds rank<K input masking and C2 pooled-context masking, but does not implement the full K-width two-view request/row-map/descriptor protocol here. Do not treat the route-decision Pass 3 acceptance as separate acceptance or completion of this model-view contract; its remaining implementation and source review are outstanding. |
| Plan C route decision/selection | `8567fd92d4dc9e214e8502f2249bb02203ac8c03`; `docs/plans/gpu-resident-route-decision-slice-20261025.md`; `ac5847d65de2af5ccb17d6a1bc109dceefeab71c5d707253980570020c1e885a` | Accepted candidate and current integrated head `codex/plan-c-k-prefix-repair-20261026` `61d3731e1531825ee7524065a158ba4765da0a84` | Independent exact-head Pass 3 **ACCEPT** for `61d3731…` against plan SHA `8c6c67ff…`; candidate equals `main` and `origin/main`. The rank<K policy-prefix plus continuation mask now bounds model inputs, zeros excluded rows before C1/C2 inference, and masks C2 pooled context; route feasibility remains separate and combines with the prefix for selection, preventing excluded/infeasible rows from being selected. Device rank order is mapped back to candidate identity before the sole compact selected-action host handoff; candidate/route evidence materializes after simulator dispatch. This closes the prior K-prefix/C2 context review finding. Scope is the three approved files `meco_candidates.py`, `meco_track_training.py`, and `meco_tracks.py`. Static source review only; no tests, device parity, simulator/replay/dashboard validation, or qualification is established. The older `dd6` implementation remains historical; this accepted slice does not complete the full Plan C route/command integration or the whole GPU-only objective. |
| Action/macro command/status handoff | Prior plan `dd278c41fa0ed8e8bb6e0097a322ca05ebbc2282`; `docs/plans/meco-device-action-macro-boundary-20261025.md`; SHA-256 `b6b4dbd5fcfca0e91e7ad079c8f172a3e2202113220e8114909ca8ee3b687774` (Pass 1 + Pass 3 accepted). New source-derived route-matrix amendment `fb89926ea0a72e76f0687b02dbf8d15a58930100`; same document; SHA-256 `27da6ff3dd2fb37c220c0f0bf074ac797b28632f9f252b5e564e1e78369001d5` (Pass 1 + Pass 3 accepted as planning-only) | No source implementation head pinned | Retains the one common device command and proposed producer/consumer paths. The amendment records base action IDs 0–7 and C route-row gathering, but route-required action classes, accepted per-class codes, `route_override` applicability, full Plan A query-to-source-row binding, neutral no-route value, and numeric native failure status remain unresolved. The amendment passed Pass 1 and Pass 3 as planning-only. Its unresolved semantics require owner decisions before implementation. No source integration or implementation authorization is added. |
| Defense simulator command-consumer seam | Proposed simulator/gamepiece owner amendment `d4b3053e3f682a3564537366b5125a21f84a7051`; `docs/plans/gpu-simulator-gamepiece-physics-tranche-20261026.md`; SHA-256 `410ed6823c02b5c27dee9f02bb900c6b8b9ce796164753c9e40fbcebcb72826f` (Pass 1 + Pass 3 accepted as planning-only) | Planning candidate only; not an ancestor of current `origin/main` | Adds only the proposed `tensor_defense_step.py::TensorDefenseStepMixin.step` command-consumer seam. Code remains blocked on final owner decisions for route-required classes, status/no-route semantics, row binding and action/macro interface alignment. It authorizes no implementation, 3v3 consumer, command producer, PPO writer, or other path. |
| Five-track PPO/rollout residency | `6e1e6981fc868f1a956f9c55fad5bd7a0947b726`; `docs/plans/gpu-five-track-ppo-rollout-candidate-residency-expansion-20261026.md`; `076c6c062e48b2cb65fbe1c0d80a75218dc32a72bec43d5aeb62d8421262056e` | No implementation head pinned | Plan text accepted, including durable generation ownership. PPO batch, rollout, reset, pressure/anchor, snapshot/resume, and numerics implementation remains required and must be reviewed as a coordinated hot-path series. |
| PPO V3 schema-only source preparation | Amendment `e0e3929c6b2e8f9109f77c1dc1105d0b367cffd7`; `docs/plans/gpu-five-track-ppo-batch-v3-source-preparation-amendment-20261026.md`; SHA-256 `248087999f2f1948b419ea67e012533794f4010f24b608b790f39c81c24833f8` | No source shell pinned | Pass 1 accepted; exact-head Pass 3 pending. The schema shell remains blocked until all V3 field-table decisions are source-bounded and accepted. After that, scope is limited to a declarative tensor schema in `meco_ppo_batch.py`; no PPO producer, compute, validation, checkpoint/RNG writer, or runtime operation is authorized. |
| Simulator gamepiece GPU path | `3d1f93f11936dcc79c70f5ae587508cfe8198257`; `docs/plans/gpu-simulator-gamepiece-physics-tranche-20261026.md`; `df12eacc11fb1d1d2c91eea627c77becfe7221f954f1105f14d05b0a5ef2afaa` | `c26244dc038a4ad044d2c730dc7e9cf8750d497a` | Plan accepted; source candidate is partial fixed-shape/masked preparation only. The `passed.any().item()` RNG-sensitive branch and other simulation residency work remain unresolved. Not full GPU simulation or parity evidence; exact-head Pass 3 is required before integration. |
| Gamepiece pass/ferry RNG v2 | Pass 1 accepted amendment `1ca7646672a39f0841db5760f1c2dbf8aa465180`; `docs/plans/gamepiece-update-fuel-actions-rng-contract-20261026.md`; SHA-256 `953f1c31436d82523b90537ccaaf1df9ebbc7bc912f7cbc07343a09fbf0751a4` | No RNG v2 source candidate pinned | Plan-only acceptance; no code authorized. It defines the `meco-gamepiece-philox-v2` per-world address/state contract. Before source migration, PPO and supervisor owner amendments must own and allowlist checkpoint serializer/resumer, V3 RNG identity/counter fields, training source/protocol binding, evaluation manifest and qualification protocol, and pass their own Pass 1/Pass 3. Keep v1 artifacts historical and do not claim v1/v2 RNG continuation parity. |
| Dashboard human-display reader | `a282b04286a7deb266a7ba9de6a88f945629477d`; `docs/plans/dashboard-v2-human-display-reader-20261026.md`; `ae946adb0dfff63bc875a98017444c1465f9d62ec35fa5d9fac8a78c88097321` | No reader implementation pinned | Accepted plan is display-only and waits for the exact PPO envelope. It must not become a metric producer, cache, gate, or provenance authority. Restart/currentness and fresh replay are post-release runtime evidence. |

This inventory distinguishes accepted plan text from implementation state. In particular, Plan A `15f7c5b` is partial static preparation only, while its status-integrated completion was rejected; the accepted action/macro plan has no code head. Its source-derived route-matrix amendment `fb89926` passed Pass 1 and Pass 3 as planning-only; it leaves route-required action classes, accepted per-class codes, `route_override` applicability, Plan A route-row binding, no-route encoding, and native failure status unresolved. ABI route-status amendment `6d46846` passed Pass 1 and Pass 3 as a planning amendment; defense-consumer amendment `d4b3053` passed Pass 1 and Pass 3 as planning-only. Their integration remains blocked on matching owner decisions and pins. The amendments leave HIP status granularity and V3 fields unresolved; they confer no implementation authorization. PPO V3 shell/row work remains blocked. The unbound diagnostic evaluation is not a Stage 1 pass. No training qualification, device/runtime execution, seeded simulator validation, dashboard restart/currentness check, or replay generation is authorized by these planning statuses; those gates remain deferred until the user manually stops training and both WX9100 leases are independently verified released, followed by the applicable exact-head reviews and approved test/evaluation protocol.

### PPO V3 contract/profile proposal review status (2026-10-26)

The following two proposals have independent exact-head Pass 1 **ACCEPT** review
and are pushed to their named `origin/codex/...` branches. Their review status
adds planning input only; neither is an owner adoption, a revision of the
immutable shared ABI, or authorization for schema/source implementation,
tests, compilation, runtime, device, simulator, dashboard, checkpoint,
campaign, training, or evaluation work. Both pin source base
`e37b2f30b6bc51b74dd72bf039e94a174ec51bf2` (tree
`8edb9998d7e3e361e83938a345c65d4fef488e1d`):

| Planning artifact | Exact accepted/pushed head and artifact SHA-256 | Current effect |
|---|---|---|
| Unified cross-owner contract decision amendment, `docs/plans/gpu-five-track-ppo-v3-unified-contract-decision-amendment-20261026.md` | Commit `5bc5a67886c7cebaf21454b5618b9649c21bff76`; SHA-256 `6c875e448ac0a88f71f497881fdc1b3a10f197ac909b5176c4c6a896734cec30`; branch `origin/codex/ppo-v3-unified-contract-amendment-20261026` | Reconciles source findings and accepted proposals into one recommended contract. Full per-policy/oracle route and path retention is preserved as a recommendation; the current immutable ABI remains unchanged. The proposal explicitly requires owner adoption and a new immutable ABI/profile revision before a schema-only declaration can be planned. |
| ThreeVsThree profile freeze proposal, `docs/plans/gpu-five-track-ppo-v3-profile-freeze-proposal-20261026.md` | Commit `fd027f91ae45ca86606a17cd54b47e7e58a12a07`; SHA-256 `8bf27df729c62a49a605bc6789de9b12529da4c00c69a8ce84549b208353a998`; branch `origin/codex/ppo-v3-profile-freeze-proposal-20261026` | Proposes profile `meco-ppo-v3-3v3-f504-n1-s6-r32-t12-v1` (`N=1`, `S=6`, `R=32`, `T=12`, `E=504`, `K=16`) and source-derived shape recommendations. It remains a proposal and does not freeze production configuration. |

The exact remaining gates before even a declarative `meco_ppo_batch.py`
schema may be separately planned are:

1. Environment, PPO and evaluation owners adopt the ThreeVsThree `E=504`
   profile and its field/raster/grid derivations, or provide a replacement
   profile ID and values.
2. Plan A/PPO/ABI owners accept fixed graph/support shapes, static versus
   per-row graph capture, and the memory implications (default source geometry
   is 12,096 nodes and 96,768 support rows).
3. Plan C/PPO owners accept `K=16`, raw feature ordering and widths `20/40`,
   ordinary-only context pooling, the continuation map, and 537-row source-axis
   binding.
4. Route/PPO/ABI/evaluation owners accept 9,681 route rows per transition,
   full oracle path retention, 152 chunks of at most 64 rows, and the 172.5 MiB
   rollout path-storage floor.
5. Observation/native owners accept target-handle and observation-snapshot
   layouts, generation allocation/reset points, stale-snapshot behavior, and
   generation-exhaustion/no-wrap failure.
6. PPO, simulator, action/macro and route owners adopt the V3 status registry,
   atomic no-op behavior, actor/critic eligibility, terminal/truncation and
   bootstrap semantics.
7. Gamepiece/score/PPO owners define the event code registry and a
   source-backed fixed batch capacity; the existing 32,768 score-ledger limit
   bounds only its episode recorder.
8. PPO/RNG/ABI owners adopt RNG streams, algorithm and draw schedule, digest
   coverage/encoding, device production and checkpoint representation.
9. Artifact/PPO owners accept the envelope header and calculate the exact
   maximum payload bytes from the final field table; no byte-tensor capacity
   is authorized before this calculation.
10. Each affected owner records accept/reject-with-replacement/not-applicable
    against exact source evidence; the shared-ABI owner publishes one
    immutable ABI revision and profile hash; every owner plan pins those exact
    hashes and passes cross-owner Pass 1 review.

The only product choices identified by the unified amendment are whether
`fuel_count=504` is the intended initial production profile, exact candidate
equivalence/denominator/empty-reference semantics if a recall metric is
required, and reward/terminal estimand only if it differs from the curriculum.
The remaining field widths, sentinels, capacities, RNG/digest encoding and
envelope details are engineering/owner decisions; the proposals' recommended
values do not become normative by review acceptance. After the gates above,
the existing schema-only preparation plan still needs its own applicable
exact-head review; PPO producer, compute, validation, checkpoint/RNG writer,
and runtime remain separately scoped and reviewed work. No owner adoption,
implementation, or Stage 1 evidence is claimed here.

### PPO V4 successor proposal review status (2026-10-26)

The new shared command/PPO transition ABI V4 successor proposal received
independent exact-head Pass 1 **ACCEPT** as a planning artifact:
`docs/plans/gpu-shared-command-ppo-transition-abi-v4-20261026.md`, candidate
commit `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b`, artifact SHA-256
`0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6`.
Its planning base is `5bf923a62f1ed0f07cbd2a6901cbf2cd41632bd4`; it is a
versioned successor proposal and leaves the accepted V3 ABI immutable. Pass 1
acceptance records planning quality only: it is not adoption by environment,
Plan A, Plan C, route, observation/native, PPO/RNG, gamepiece/score,
artifact/manifest, or evaluation/curriculum owners, and it authorizes no
schema or source implementation, tests, compilation, runtime, simulator,
dashboard, checkpoint, campaign, training, or evaluation work.

Before any schema-only plan can be proposed, every affected owner must
explicitly adopt this exact V4 artifact hash or publish a versioned replacement;
the shared-ABI owner must then publish one adopted immutable ABI/profile hash,
and each owner plan must pin and independently review it. The V4 document's
proposed profile, capacities, target-handle lifecycle, field table, route
retention, digest/RNG encodings, payload and envelope remain recommendations
until those gates pass. The training objective's Stage 1 and uninterrupted
five-track continuation evidence remain separate runtime gates; this planning
acceptance establishes neither.

### Refreshed dependency order and campaign boundary

1. Keep the active two-WX9100 campaign and its live checkout, supervisor,
   inputs, outputs, checkpoints, and leases untouched. Training may continue
   uninterrupted until the user manually stops it. The available diagnostic
   evaluation is unbound and cannot count as a Stage 1 pass. Update counts,
   lane completion, old checkpoints, and stale replays are not substitutes for
   a source/protocol-bound Stage 1 result.
2. While the campaign owns the devices, allow only isolated documentation and
   source-preparation work explicitly permitted by an accepted Pass 1
   amendment. Do not import, test, compile, initialize/query a device context,
   run simulation/evaluation, or restart dashboard services under this
   sequence. Preserve the user's dirty
   `frc_defense/csrc/tensor_collision_multi_hip_kernel.hip` file.
3. After manual stop and independent verification that the supervisor and its
   workers are terminal and both WX9100 leases are released, integrate in this
   order: shared ABI; Plan C route decision and K-capped view as a coordinated
   device-query/model-input tranche; Plan A support and decision against that
   shared route API; PPO rollout/batch/reset/snapshot/numerics as one reviewed
   training-hot-path series, with the separate PPO and supervisor owner
   amendments for RNG state/checkpoint/evaluation protocol binding reviewed
   before any gamepiece RNG v2 source migration; then action/macro handoff
   against finalized A/C actions and PPO replay contracts. Integrate the
   gamepiece v2 RNG and simulator source only after all required owner
   amendments and exact-head reviews. Keep every patch inside its approved
   path allowlist and preserve v1 artifacts as historical-only.
4. For every implementation patch, resolve current base and candidate full
   hashes, inspect the approved paths, preserve semantic/provenance invariants,
   obtain independent exact-head Pass 3, then integrate those reviewed bytes.
   A changed base, candidate, plan artifact, or path set invalidates approval.
   Accepted plan text alone never authorizes source integration.
5. Only after source review/integration and lease release, run approved
   device-residency, parity, determinism, snapshot, safety, and performance
   gates on the two WX9100s. Then run seeded end-to-end simulator evaluation
   of changed behavior, inspect its fresh record/replay, restart and verify
   current dashboard backend and frontend, and generate a fresh dashboard
   replay. Record scenario, seed, replay ID, source/checkpoint/protocol
   provenance, service identity, and served asset hashes. Do not use the P600
   or CPU numerical fallback.
6. Close Phase 0 authority and Phase 2 data/warm-start artifacts, then bind
   current-source checkpoints and a non-diagnostic five-track protocol.
   Stage 1 passes only when all five A1/A2/A3/C1/C2 rows satisfy the curriculum's
   paired workload, confidence, safety, replay, and dashboard gates. A ready
   preflight or unbound diagnostic evaluation is not a pass.
7. After independently verified Stage 1 completion, continue all five tracks
   with healthy evaluation/checkpoint cadence and no starvation or incomplete
   cycles. Continue until the user manually stops training. Qualification,
   promotion, and remaining full-plan experiments are separate later gates.

### Pass 1 review checklist

- Confirm the patch changes only this sequence document and record its exact
  blob SHA-256 in the review result.
- Confirm each pinned commit and plan artifact SHA-256 resolves to the stated
  local object and exact bytes.
- Check that accepted plan text and partial source candidates are distinct;
  plan acceptance is not implementation, source integration, runtime
  validation, or Stage 1 success.
- Confirm source integration requires independent Pass 3 on the exact head
  before merge, and that a moved base/head invalidates review.
- Confirm training remains uninterrupted, only the two WX9100s are in scope,
  the dirty user HIP file is preserved, and runtime/dashboard gates wait for
  manual stop plus verified lease release.
- Confirm unbound diagnostic evaluation and progress counters cannot satisfy
  Stage 1; seeded simulator replay and dashboard backend/frontend currentness
  remain explicit post-release gates.
- Confirm the amendment does not claim implementation or achievement of the
  five-track objective.

## Historical pinned candidate inventory (superseded where amended above)

Remote heads were freshly read from `origin`; hashes below identify immutable
observed heads, not approval. An implementation head is **pending** until its
exact tree is reviewed in Pass 3 against its approved plan and then-current
main. Existing plan branches do not authorize merging their accumulated
history wholesale. Pass 2 must inspect each diff, select only in-scope commits,
rebase/cherry-pick as needed, and publish a new integration head for review.

| Area | Pass 1 plan / source contract | Candidate or integrated head observed | State and dependency |
|---|---|---|---|
| Plan A device decision/support | `docs/plans/gpu-resident-plan-a-decision-slice-20261025.md`, SHA `21b5610319116ee80e95cdcc668c4caa3491dd8b2a204e7fdf71bd8366f87235` | integrated source head `daa2663fc2fecdb183c304bc4dff41bcba9a1acb` on `main`; prior reviewed candidate `d656a8d8f320f746e2365ce7530177fa7daa233d` | Final exact-head Pass 3 **ACCEPT** for rebased head `daa2663…` against plan SHA `21b561…`; it replays the reviewed two-commit, two-path correction (104 insertions/12 deletions; diff SHA-256 `29dfda742a9274d75b21c41408bc5673a679af1a6481fee74a93cbdc35636ce3`). Canonical route cost units fail closed, invalid chunks invalidate the merged result, and selected route/cost evidence remains bound to source identity and PPO replay. It is integrated in current main. This is source review only and does not establish device/runtime parity or qualification. Preserve A1/A2/A3 all-node support, fixed 64-row chunks and stable global mapping. |
| Plan C route decision/selection | `docs/plans/gpu-resident-route-decision-slice-20261025.md`, SHA `8c6c67ffb2934fa3a4ce40b54c5cc2f2e2cb4877b5eedac9c31a2936a6879b5a` | rejected predecessor `codex/plan-c-dispatch-boundary-repair-20261026` `8befd2df92a93d9e19c9c721952ab7b645eedebe`; fresh repair `codex/plan-c-k-prefix-repair-20261026` `61d3731e1531825ee7524065a158ba4765da0a84`, based on current integrated main `daa2663fc2fecdb183c304bc4dff41bcba9a1acb`; earlier rejected heads `5e59083f030ac4385b2dc367a88b48effc7db46c` and `a70cd1380451167f50f33c45cde131d753e0ae7a`; original no-op integration head `e87d757cb70262b523a4342315695bf600cf24c7` | Prior exact-head Pass 3 rejection `8befd2d…` required that full extra-sensor and approach-offset rows not reach C1/C2 and that C2 pooled context exclude inactive rows. Fresh Pass 2 candidate `61d3731…` uses a fixed-order device `model_input_mask` for rank<K rows plus continuation, zeros excluded feature rows before scoring, uses the mask for C2 context pooling, and keeps route feasibility separate; exact-head Pass 3 was pending in this superseded snapshot; the current inventory above records the later independent acceptance. It touches `meco_candidates.py`, `meco_track_training.py`, and `meco_tracks.py` under the accepted plan. Do not use this historical snapshot to infer current integration state. Prior reviews successively fixed post-dispatch `route_trace` provenance and dynamic eligibility `.item()` synchronization, then found the fixed-K offset coverage issue; those are not the current blocker. The earlier integration head `e87d757…` is already an ancestor of source baseline `160911abe…` and current main, so merging it is a no-op; this was the pre-repair source state; the current inventory above records the accepted K-prefix repair without claiming full Plan C or whole-compute completion. |
| Candidate membership/fallback | `docs/plans/gpu-candidate-eligibility-fallback-tranche-20261026.md`, SHA `6edd6c81bdd92c96c82bb0a3a85d5949ac953a3df1c8729fadb333e30ca4add1` | plan `8554fe3ef9040306c02f87d66aa7ee49c8cbcdeb`; implementation `d455811fa3dbb44b875715fa022e5b8f0eb8a34b` | Implementation head is already an ancestor of planning `main`; do not remerge. Verify exact source/provenance, then integrate with the still-pending Plan C API. This does not prove complete candidate/route residency. |
| Five-track PPO batch/residency | `docs/plans/gpu-five-track-ppo-batched-residency-expansion-20261026.md`, SHA `61613da9f120d3ee22b53b9a09de529ea6981e3639bd43a1fcea966654a85b0b`; dependent rollout/reset/pressure/snapshot expansions below | `codex/five-track-ppo-batched-residency-impl-20261026` `57e977496c364a82d28481d949d3d33900f13499` | Pending exact-head Pass 3. This branch adds `meco_ppo_batch.py` and changes the training hot path; coordinate one ordered integration with all reset/rollout amendments, not parallel merges. No PPO or sampler semantics may drift. |
| PPO rollout/candidate replay | rollout plan SHA `835e402398db64fd0ed2b3ac17d24d3e7567ce322c25bccdb7d543656bdb92b4` at head `7ce8dd894e9beff41146543264f44fe420187963` | same plan branch; implementation dependency is PPO candidate above | Required to keep selected action, log-probability, masks and replay row aligned for all five. Integrate with PPO batch work. |
| PPO selected-row reset | SHA `ad90181efcdafa141b6f1006ac7017a53e9b17204c3045d8556a7f011d338fb5` at head `e7e426c556b80b7eda8bcefae014bf3c29c9129d` | plan branch | Recurrent/reset semantics must be fixed before qualification or resumed training; retain per-row reset behavior and seed progression. |
| PPO pressure/anchor/reset expansion | SHA `c5c5feafb808e495ac074f6f7782d6ee39c39a0024d0ff59518593b0a51ed2af` at head `d5bb8856399edf9b8119a0a93e2963a30ec1f459` | plan branch | Integrate only after the common PPO residency API is reviewed; preserve pressure phase and valid anchor/reset masks. |
| PPO snapshot v2 | SHA `f7a9859ee9ad7436f8e9821f6c6ee0beebad37d14bc2c7ac1bb1ce5ef645f94c` at head `f3d8ddcbf1bbb1265de35682045e531fb6aa3dbc` | plan branch | Requires rollout/PPO state contract. Checkpoint schema migration must be explicit and source-bound; old diagnostic checkpoints cannot become current-source evaluation inputs. |
| PPO numerics | expanded plan head `39c0466cafca4c3fd93848c019bedbca08fb8a3b`; base numerics plan head `d843d80196e7ee77d6b5b61689da56ecab0f5786` | planning branches; no standalone implementation assumed | Integrate only against batch/residency API. Keep loss, entropy, clipping, minibatch order, RNG and optimizer math consistent and device-resident. |
| Training action/macro boundary | `docs/plans/meco-device-action-macro-boundary-20261025.md`, SHA `6baf32f4356e3712c8e8c216c5a0fd69a20e95b693ecf8000c8c29d146fad7a7`, head `3399ef1f195625d8a2ba7c9ec68f717bd5191bdb` | plan only | Approved planning dependency, not implementation. Amend upstream A/C and PPO chosen-action contracts to forbid dynamic numeric D2H. Implement after those selection and batch contracts land; it touches the same `meco_track_training.py` hot path. |
| Observation/sensor residency | plan head `2d67e108da7dd99ddf353cd5d5d25261c48abc06`, source implementation head `da6a2223d9822d87088e7cf90ad108c3a7384e29` | implementation head is already an ancestor of planning `main` | Do not remerge. Verify its exact-head review/provenance and GPU fuel-physics dependency before claiming closure. Mainline presence is not runtime proof. Preserve truth versus policy-visible observation separation; no host numeric decoding or snapshot assembly. |
| Simulator gamepiece physics/reset | plan head `027f694c4c729c313f48440475907164d985398c` | implementation head `4c604e0f30324d0b20bb38fe5f9b177264a5adcd` is an ancestor of planning `main`; execution-status ledger records reviewed/integrated | Do not remerge. Existing integration is source evidence only. Keep its fixed-shape, fail-closed and graph-capture conditions. After device release, verify parity; no claim of all simulator physics until call-graph audit. |
| GPU score-event buffer | `docs/plans/gpu-resident-score-event-buffer-20261026.md`, SHA `3095cbee304f21d1d84ae91c1152722f955df47fba58f2399d3e6302d518eff1`, plan head `47ff1037dc5bf0035b5a1d4523f5bddae240351e` | implementation `82009ecdd7b827024928f2ded0ce56ba9889c863` | Pending exact-head review/integration. The current status explicitly says qualification fails closed when event extraction needs host materialization. This boundary depends on simulator score semantics and may not bypass score-estimand approval. |
| v4 GPU bootstrap | plan SHA `48db0d260fe29da4a55d65c553746d4aa0813237a6bb10d5eabd52d285340ca9`, plan head `0b187e6288a084e109056ca7c1aec908c3d0c9e5` | implementation `9497615c9cbd355b8b065e9ccd9b7ead87ad00be` is an ancestor of planning `main`; status ledger records integrated | Verify ancestry/provenance only; do not remerge. It supplies one statistical component, not calibration, margins, qualification, or all-metrics GPU residency. |
| Phase 0 freeze, tuple partition, approvals and paired calibration | use current source-bound Phase 0/paired-calibration plans, including `phase0-effective-freeze-partition-plan-20261025` head `7e7e7253c9756ef17bc6b33582921e19cd645556`, baseline calibration head `a5aebfb034e0329ed66f097ea38d8cf4e93723ee`, paired calibration head `268841060593f3ea21e2409b61c88a0134eb923a`, and integrated source at `86706ad9d355138a43b48f2578c0f575b05efd4c` | substantial source is integrated; the named plan/source heads are ancestors of planning `main` | Hard training/evaluation gate: authentic effective config receipt, freeze v3, closed disjoint partitions, complete paired D0/D1/D2 schedule, human estimand approval and human margin approval. No generated artifact may substitute for human approval. |
| Phase 2 collection/fitting/warm starts | runtime plan head `054a21ea0fce42100a8c5eb5993d2e5c9e0bf53b`; warm-start execution plan head `eed7a2f1d7a35ec9b89b49c547837ef492b6749c`; implementation branch `79e2d228cde22932b8cf437337be8c1de7859637` | source components partially integrated; complete train-case artifact set not established | Requires Phase 0, all required cases, A1/A2/A3 U+B+P and C1/C2 shared source campaign artifacts, source/legal/protocol binding, manifest closure and checkpoint compatibility. No production collection/fitting until manual stop and both WX9100s released. |
| Stage 1 preflight | `docs/plans/five-track-stage1-readiness-preflight-20261025.md`, SHA `68decf57abac362b83d24a86b450a59d4ac361c13d9744c022a5dcae8db0edfa`, exact approved plan head `214fe0bf564e137d0114a07df53d7dd250bc3cc0` | integrated source head `bfeb1294c99f3d656a9f53f20077cbdb8dcdc7cc` on `main`, an ancestor of current `main`/`origin/main` `61d3731e1531825ee7524065a158ba4765da0a84` | Exact-head Pass 3: **ACCEPT** against plan `214fe0bf…`; the nine-commit series was replayed onto the then-current main, preserves the same five allowlisted paths, and its cumulative diff SHA-256 `336a7003086202f6d33101a47d1be1751a8ff0b3e1795e7a9ca1964707c50d59` matches the previously reviewed source patch. It is integrated in current main as the `bfeb129…` ancestor. The implementation adds the plan-approved metadata-only readiness preflight and does not establish readiness or a Stage 1 pass; do not run it on live artifacts/checkpoints during the active campaign. Source integration baseline remains `160911abe1cf4644628053dae054e363b7fb010b`. The sequence first landed plan-only at `3861b56b24d9e88dedd4b6fbf3458162b72177ca`. The five-path allowlist is: `docs/MECO_QUALIFICATION.md`, `frc_defense/meco_prediction_contract.py`, `frc_defense/meco_stage1_preflight_contract.py`, `frc_defense/meco_qualification.py`, and `frc_defense/meco_tracks.py`. `docs/MECO_FULL_PLAN_EXECUTION_STATUS.md` remains excluded until after integration and its own review. This is static contract evidence only. |
| Stage 1 qualification/gate | pickup qualification plan head `020aaa1283e4e9b407e95b943f999cdd658d3463`; curriculum stage gate head `feedc67cf810dee7156d831c5c604187032c0494` | planning only | Qualification requires exact current source/checkpoint/curriculum/protocol binding, all five rows A1/A2/A3/C1/C2, preregistered paired baseline, Stage 1's full case count and confidence/safety gates, fresh simulator records/replays, and no diagnostic split. |
| Qualification coordinator/dashboard evidence | coordinator plan SHA `94f5119af2401298ea73738580f84a183953033eef308bccf15b748426f3ebb2`, head `8c73d4273d164394075c1b28a04bd190830f4292` | plan only | Integrate coordinator only after qualification result contracts are fixed. Dashboard currentness is a required evidence step; dashboard source changes/restart must be isolated from active services and approved separately. |
| Phase 3 Plan A PPO/replay contract | plan `c9df7752400b3f08eb8a878bc324455f23488822`; implementation/test candidate `c51b081a6c12dee42ff01f86640da3258df647c6` | plan is in main; implementation head is not an ancestor of planning `main` | Re-resolve its exact path allowlist and review evidence before integration. Run no tests in this plan. Eventual tests must stay GPU-free unless separately approved and must include the required simulator/replay gate for behavior claims. |
| Phase 4 adversarial/runtime gates | direct pursuit plan `c629f1228060bed97119bea6b777a7e6cc40327c`; focal pose `2357ae4a9661c7cb6d6c038a72881a55fbdcfbec`; trench `6d69711a892ff84928d2c805bd7b1a5567bd8eca`; reactive pool `6936feeb4b2843d8964a8db5b0fd5d76ee1a4f03`; start-profile executable plan `a9fdc9387a3cd8557f005863b4517781ef38ba9a` | these plan heads and current implementations `40fbf7d4529594bb5575702209ae5c56638b1c3d`, `65bc56e986facc19949852b6c942ccc8bd09179f`, `ceaac65dcc7cbe1a5217c6fae7b3ac0548d36edf`, and `b580d13d139e65a43568b4384a27bb99c783388a` are ancestors of planning `main` | Do not remerge. Status says fresh outcomes for multiple factors remain unavailable/non-authoritative and the 14-category requirement is not closed. Keep this as a post-stop seeded-simulator evidence gate; each changed behavior needs current-source replay and dashboard verification. |
| Phase 4/5 status and deployment evidence | status/dashboard source `637cc4b64fd5d1b22532305dd86fc14deb66a913`; Phase 5 capture plan `cb4ac110f963a9055d595d587d79d3475b13dc16`, implementation `7f374320011ce2579893b23adf190610e1a6740d`, dashboard-currentness `c4731551046699a03be04116351df99dfea7049d`; v4 gate plan `0b187e6288a084e109056ca7c1aec908c3d0c9e5`, implementation `9497615c9cbd355b8b065e9ccd9b7ead87ad00be` | listed implementation/currentness heads are ancestors of planning `main`; main status records runtime/deployment evidence gaps | Do not remerge. These are full-document gates, not prerequisites to declare Stage 1 training continuation if the approved curriculum defines otherwise. They are mandatory before claims of adversarial generalization, five-track winner, production deployment, profiling or promotion. Defer simulator/Hailo/device execution until release and applicable gate approval. |

The abbreviated candidate handles above are for orientation only. Pass 2 must
re-resolve every full branch name and SHA from `origin`, calculate ancestry and
diff from then-current main, inspect its Pass 1 artifact SHA, and record the
exact new integration head. A changed remote head, branch deletion, unseen
review correction, or additional source plan is a mandatory stop and plan
amendment. Do not merge unreviewed side-branch history merely because it is
reachable from a candidate branch.

## Integration order and pass boundaries

Each numbered tranche is its own reviewable patch. **Pass 1** is this sequence
plus the referenced narrow plans. **Pass 2** takes only the listed files and
source commits, rebases to current main, and pushes a candidate. **Pass 3** is
an independent exact-head review against the source plan, source spec and the
prior integration head. A rejected Pass 3 requires a new Pass 2 correction and
another exact-head Pass 3. Only after approval may the patch be integrated.
Status, dashboard provenance and execution-status changes are separate
patches and themselves require exact-head Pass 3 before integration.

1. **Reconcile the source plan and evidence baseline.** Verify current main
   still has the pinned plan and gate inventory; audit what is already
   integrated by ancestry and source diff. Do not rewrite the normative plan
   or mark stale summaries current. Produce a scoped integration manifest
   mapping every source-plan requirement to current code, plan, evidence or an
   explicit remaining gate.
2. **Complete source-bound authority (Phase 0), then Phase 2 artifacts.**
   Finish only missing reviewed code contracts first. Human-authored score
   estimand and operational margin artifacts must be separately supplied and
   approved; they cannot be authored by an integration agent. After manual
   training stop and device release, materialize/validate the effective
   freeze, four-way partition, paired controls and five-track assignment;
   then collect/fix/train all five source-bound warm-start artifacts. Keep
   these as distinct protocol/data/runtime tranches and Pass 3 each.
3. **Integrate Plan C candidate/route/selection and deterministic D1/D2
   contracts**, then **Plan A all-node/support/route/selection** against the
   shared device route API. Include the five required tracks in the same
   integration matrix: A1 U-Net+GRU, A2 structured SE(2) U-Net+regional
   attention+GRU, A3 hierarchical SE(2) GNN+regional attention+GRU, C1
   candidate MLP+GRU, C2 map/candidate cross-attention+GRU. Keep D0/D1/D2
   matched controls and prediction modes explicit. No track may silently
   change input privileges, candidate recall, action support or safety gates.
4. **Integrate common five-track PPO and selected-action/macro boundary as one
   coordinated hot-path series:** batch residency and numerics; rollout,
   action/log-probability/replay row binding; selected-row resets; pressure
   anchors; snapshot v2 and resume determinism; then Plan A/C action and goal
   tensor handoff/macro state. Amend upstream “compact selected action”
   allowances first. This order prevents two concurrent implementations from
   rewriting `meco_track_training.py` with incompatible row, reset or replay
   contracts.
5. **Integrate device observation/sensor, simulator physics, action remasking,
   collection/scoring and score-event recording** against the reviewed state
   contract. The fixed-mask gamepiece code already on main remains a scoped
   prerequisite, not proof of complete GPU simulation. Stop if any changed
   call graph still materializes numeric world state, decisions, rewards,
   scores or terminal states on CPU; write a new Pass 1 amendment for any
   kernel/path outside the accepted allowlist.
6. **After source code has exact-head reviews, validate offline contracts and
   hardware behavior in separate passes.** Pure/static checks may be run only
   when the reviewed tranche explicitly allows them and must not import code
   that initializes or probes a device while the leases are occupied. After
   the manual stop and independent release gate, execute device residency,
   parity, determinism, snapshot/resume, safety and performance tests on both
   physical WX9100s only. No P600, single-device fallback or CPU numerical
   oracle may satisfy production acceptance.
7. **Run seeded end-to-end simulator evaluation** on the changed behavior only
   after device parity passes. Inspect and retain simulator record/replay and
   its source/checkpoint/protocol/seed provenance. Restart the applicable
   dashboard service from the exact reviewed checkout, verify process source,
   APIs and served frontend hashes, then generate a fresh replay and inspect
   it. Do not reuse pre-restart or old-source replays. Follow `AGENTS.md`.
8. **Complete Stage 1 readiness and qualification** only after valid five-track
   current-source checkpoints, protocol-bound split/assignment, preregistered
   matched D0/D1/D2 baselines and calibration/margins, Stage 1's full paired
   workload, safety and confidence criteria, replay inspection, dashboard
   currentness and all required reports validate. A `ready` preflight is not a
   pass; diagnostic run counts, optimizer updates, stale checkpoints, status
   JSON, model inventory or synthetic unit fixtures do not count as Stage 1
   evaluation results.
9. **Continue training after the Stage 1 gate** with all five tracks retained
   and no starvation or incomplete cycle; the gate and supervisor must show
   ongoing protocol-bound training and healthy checkpoint/evaluation cadence.
   Preserve the user's instruction that training remains uninterrupted until
   manually stopped. A later evaluation failure may report failure but cannot
   request stop/restart or relaunch by itself.
10. **Finish the remaining full-plan experiments and deployment gates**:
    held-out adversarial/generalization runs; A1/A2/A3, C1/C2, D0–D3, P=0,
    K=8/16/32, matched-prediction and inference-disable/fallback comparisons;
    uncertainty/route latency and safety metrics; capture/ONNX/Hailo support;
    p50/p95/p99 end-to-end profiles; final source-bound qualification,
    promotion and rollback evidence. Keep “Stage 1 passed”, “qualified” and
    “promoted” as distinct states.

## What can integrate during the diagnostic campaign

The active campaign must remain unchanged until the user manually stops it.
While its separate diagnostic checkout/process owns the WX9100s, the following
may proceed only after exact-head Pass 3 and only as isolated source changes on
the integration checkout/mainline:

- planning and source/provenance documentation;
- metadata-only validators and device-independent source contracts whose
  reviewed behavior does not read, stat, hash, deserialize or rewrite live
  campaign/checkpoint payloads;
- code review and static diffs. No import or test command unless a Pass 3
  plan explicitly proves it is side-effect-free and cannot initialize a GPU,
  simulator, dashboard service or live artifact path; this plan itself runs no
  tests or imports;
- already-computed candidate source integration that does not deploy/restart
  the active trainer or alter its manifest, input, process, output tree,
  scheduler or lease. Production implementation remains non-operational until
  its post-release device/simulator gates pass.

During this period, **do not** modify the live training checkout or campaign
root, stop/start/reconfigure the supervisor, move or relabel checkpoints,
touch active outputs, inspect campaign/checkpoint payloads, run
training/evaluation/collection/fitting, invoke any device API that initializes
a compute context or can acquire/reconfigure a device or submit work, run a
simulator or behavior test, build/compile GPU kernels, restart the dashboard
or generate a replay. Read-only device-status/utilization queries are allowed
for progress reporting only when they do not initialize a compute context,
acquire/reconfigure a device, submit work, or inspect campaign/checkpoint
payloads. Missing telemetry or low utilization is not evidence that training
stopped or a lease was released. Do not run dashboard currentness actions while
prohibited by the user's active-campaign instruction. Any dashboard or runtime
evidence needed later is an explicit post-stop gate.

Integration being safe to commit does not mean its code is safe to execute.
Source integration must not be represented as runtime readiness. The diagnostic
campaign remains historical/non-qualifying and cannot retroactively become a
current-source Stage 1 pass.

## Manual-stop, device-release and training-transition gates

1. Do not request or infer a stop. Wait for the user's manual stop instruction.
2. After that instruction, independently confirm the supervisor is terminal,
   its owned child workers exited, and both assigned physical WX9100 leases
   are free. A stale status file, missing heartbeat, expired poll, missing
   utilization telemetry or low GPU utilization is not release evidence.
   Read-only status may help observe utilization but does not establish lease
   release. Never use the P600.
3. Only then run the approved device/static test sequence and source-matched
   seeded simulator evaluation; restart/verify the dashboard and generate a
   fresh replay as specified above. Report tests, simulator scenario/seed and
   replay ID, dashboard backend/frontend currentness, and remaining failures.
4. A new five-track training campaign can start only after independent reviews
   of each implementation tranche, device/residency and simulator behavior
   gates, valid Phase 0 authority and Phase 2 artifact closure, all-five
   source-bound checkpoint manifests, non-diagnostic evaluation protocol,
   explicit two-WX9100 lease admission/attestation, and a reviewed launch
   manifest. The campaign may continue until the user manually stops it; this
   plan does not authorize an earlier stop or any restart before these gates.
5. After launch, verify all A1/A2/A3/C1/C2 lanes are actually trained, evaluated
   under their assigned protocol and continuing beyond Stage 1 without
   qualification errors. The user keeps sole authority to manually stop the
   run.

## Full-document gate ledger (must remain visible)

| Source-plan section | Required integrated/runtime evidence before completion |
|---|---|
| 1. Decision and hypothesis | Matched held-out results showing whether learned tracks beat D0/strongest matched deterministic control; no winner before confidence and safety criteria. |
| 2. Five models | Current source-bound checkpoint/export per A1/A2/A3/C1/C2; device memory, exact output contract, actual operator support and simulator behavior. |
| 3. Deterministic/predictive controls | Fresh D0/D1/D2 and if claimed D3 runs with same seeds, information, candidate generator, routing, prediction access and evaluation budget; paired outcome/runtime intervals. |
| 4. Shared observation/planner | Fresh five-track replay inspection; latency, freshness, deterministic fallback and action legality under device load; no assumption that XY AD* is global SE(2). |
| 5. Training strategy | Frozen baseline/split; valid Phase 2 datasets/labels/warm starts; stochastic PPO/log-probability; all-track rollout/replay, resume/RNG and learning evidence. |
| 6. Metrics and gates | Current v4 protocol; human score estimand and operational margins; calibration, five-track qualification, safety/noninferiority, confidence intervals, and one-use final-test policy. |
| 7. A/B experiments | Results—not merely analyzers—for A1/A2/A3, C1/C2, A-vs-C, D0/D1/D2/D3 as applicable, P=0, K=8/16/32, matched prediction and inference-disable/fallback. |
| 8. Implementation boundaries/order | All five current-source artifacts, reviewed integration, representative current replay, dashboard provenance/currentness, training continuity and rollback/checkpoint identity. |
| 9. Unresolved decisions | Empirical legal-grid adequacy, candidate recall/route costs, predictor calibration/occlusion, held-out generalization, Hailo support and measured end-to-end latency; no code-presence inference. |
| User GPU-only requirement | Complete call-graph residency contract over observation, tracking, candidate/route, simulator/safety, training/fitting, evaluation/statistics and artifact boundaries; two-WX9100 traces plus proof no CPU numerical/fallback path. |

## Acceptance evidence and stop conditions for this plan

- This artifact is Pass 1 only. Pass 3 must approve its exact commit and file
  hash before any implementation or integration work is attributed to it.
- Every Pass 2 integration records: base main SHA, candidate full SHA, exact
  plan file SHA, changed path allowlist, reviewed semantic invariants, source
  identity effects, and exact pushed integration SHA. Pass 3 reviews those
  exact bytes before main is changed.
- The current execution-status document and three-pass provenance ledger are
  updated only after the corresponding source integration has landed and an
  independent Pass 3 has approved it. Each ledger/status update gets its own
  exact-head review and does not assert tests, current dashboard, simulator,
  training, device or qualification evidence that has not been produced.
- Any source path, requirement, reviewer result, candidate head, active
  campaign state, lease state, protocol contract or artifact digest outside
  this inventory triggers a stop and a Pass 1 amendment. Keep commits
  incremental and pushed, but do not push an integration to `main` without
  exact-head Pass 3 approval.
- This plan is not itself implementation, Pass 3 approval, device evidence,
  simulator evidence, a Stage 1 pass, qualification, promotion, or permission
  to stop/restart training.
