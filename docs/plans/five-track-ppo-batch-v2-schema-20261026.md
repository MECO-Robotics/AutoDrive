# Pass 2 schema note: five-track PPO batch v2

**Base:** `f4a6192537f06fbf9c0e889709e3949a680b52fd`  
**Governing plan:** `gpu-five-track-ppo-batched-residency-expansion-20261026.md`  
**Scope:** static design only. No tests, runtime, device probing, simulator, dashboard, replay, campaign, or training activity.

## Capacities and layouts

- `R = config.rollout_steps`. `MecoTrainConfig.validate()` requires a positive Python integer before allocation; R is captured once as immutable schema metadata for the update. Each update allocates exactly R slots. Slot position is the static host rollout counter; slot validity is a device `bool[R]` mask.
- `N = 8 * nx * ny`, directly from the fixed-capacity `LegalPoseGraph` contract (`HEADING_BINS=8`; `nx,ny=env.planner.nx,ny`). The graph's compact representation is padded to N with `node_valid[R,N]` marking its legal prefix. `E=10` directed neighbors per node, `G=64` coarse regions, and coarse edge width 64 are fixed in `meco_pose_graph.py`.
- Plan A's canonical support is exactly `P=8*N`: `build_plan_a_support` constructs eight descriptor columns per node and flattens row-major. `support_valid[R,P]` is the only dynamic mask. The ids/order remain `node_id`, behavior `[0,1,1,1,1,2,3,4]`, action `[7,0,1,2,3,4,6,5]`, descriptor `[-1,0,1,2,3,-1,-1,-1]` per node; padding beyond N is invalid.
- Plan C capacity is `K=config.candidate_k+1 <= 33`: K-1 policy rows and one reserved continuation row. Its `valid[R,K]` mask is authoritative. Candidate integer identity/order, action, target, goal and route override are tensors aligned to K; strings remain static/report metadata only.
- Observation input layouts follow current model contracts: raster `[R,6,H,W]`, globals `[R,7]`, hidden `[R,64]` (or a same-device zero tensor when absent; `HIDDEN=64`). All rows share H/W and dtype from the fixed environment/model instance. Value prediction uses the same `globals` and recurrent hidden state.
- A graph fields are batched with leading R: node features `[R,N,9]`, directed edges/masks `[R,N,10]`, node validity/regions `[R,N]`, coarse edges/masks `[R,64,64]`, and support/path tensors at N or P extent (`pair_node_ids`, behavior/action/descriptor ids, mask, route/risk/path cost, parent edge, traversal order, ancestor jumps, root and fallback). C payloads are candidate model features `[R,K,2F]`, raw/missing features `[R,K,F]`, validity/action/stable candidate integer ids `[R,K]`, target/goal `[R,K,2]`, heading, route/risk costs and override flags.

## Common rollout fields and mask truth table

Common numeric fields are action index, old log probability, old value, reward, episode-boundary bit, execution-failure bit, actor-valid bit, critic-valid bit, slot-valid bit, and per-row frozen input digest bytes. `episode_boundary = ended OR truncated`, preserving current update behavior (it prevents GAE from bootstrapping through either reset); the last rollout bootstrap remains the existing `bootstrap_value`. `actor_valid = slot_valid AND NOT execution_failure`; `critic_valid = slot_valid`; `slot_valid = true` only for a committed rollout slot. Padding has all masks false and inert zero payloads.

| Slot | Actor | Critic | GAE contribution | Adam transition |
|---|---:|---:|---:|---:|
| active ordinary transition | 1 | 1 | 1 | one sequential update |
| active execution failure | 0 | 1 | 1 | one critic-only update |
| inactive capacity padding | 0 | 0 | 0 | exact no-op, including moments and step |

GAE uses fixed reverse slot iteration over tensors. The numeric recurrence is masked by slot-validity; actor advantage normalization uses only active actor rows with the current population standard deviation and `1e-8` floor. The implementation must prove an empty actor set cannot change critic-only rows and does not introduce a host branch.

## Replay digest and verdict

Replay binds every active row's complete ordered numeric input, not Python row metadata. For each statically enumerated slot, serialize tensor metadata from field-name/dtype/shape schema and concatenate contiguous on-device byte views for that row's A/C payload and common transition fields, then call the existing device `sha256_bytes` kernel. Store its 32 output bytes into a separate immutable `[R,32]` digest tensor at slot write. Replay reconstructs each row's byte stream and compares all 32 digest bytes on device, then checks active-row replay log-probabilities at the existing `atol=rtol=5e-5`. Reduce schema, finite, digest, probability and padded-mask predicates into a scalar device status. A false status masks every optimizer state transition to exact no-op; host routes only a typed terminal status after the attempted update. Hex strings are passive serialization after completion.

This digest scheme must be checked against the actual HIP extension API before implementation. Any required CPU byte conversion, per-row host hash, unapproved native target, or inability to preserve an immutable expected digest is a stop/re-review condition.

## Scoring, Adam, and RNG call/dataflow

```text
rollout fixed slot counter -> typed tensor slot write -> device digest snapshot
    -> device batch replay / five-track batched forward
    -> device replay + finite/schema verdict
    -> device GAE and actor normalization
    -> for each static epoch and position: device permutation index
       -> tensor gather one transition -> existing-equivalent one-row loss
       -> backward -> device-masked sequential Adam state transition
    -> typed status + passive metric serialization + v2 checkpoint
```

`score_candidates` must retain C row order/missingness/mask/route/risk/uncertainty and C2's full candidate-query context; leading-R forwards must not mix candidates between rows. A1 must gather utility at each row's independent pose coordinates. A2/A3 must use their own fixed graph tensors and produce the same hidden state and support logits as independent one-row references. A `P=8N` gather preserves canonical support; no candidate/support compaction is allowed in PPO.

Each track has two explicit device generators: categorical action sampling and per-epoch permutations. Each is initialized from host seed metadata once on the assigned HIP device and saved with a versioned state/counter in the v2 checkpoint. Model/value initialization must consume a declared device generator after device placement; Python/CPU RNG cannot seed weights. Fixed slot/epoch loops do not affect RNG when padding is present: generate a permutation of R positions, then mask invalid slots without drawing additional random values.

The masked Adam representation uses the existing model parameter order as static metadata. For each per-transition loss, autograd produces gradients; device tensors hold `step`, first moment, and second moment per parameter. The active predicate is `slot_valid AND update_verdict AND critic_valid`; actor loss is separately masked by actor-validity. Compute standard Adam (`betas=(0.9,0.999)`, `eps=1e-8`, zero weight decay, no AMSGrad) and select old parameter/moment/step values on inactive slots. Bias corrections use device step tensors. This must be compared to ordinary sequential Adam for every active row; any required `.item()` or disagreement stops the tranche.

Checkpoint schema becomes v2 and binds batch/optimizer/RNG contract identifiers and track schema. Legacy v1 resumes fail closed. Checkpoint save/load serializes completed opaque tensors; it does not use host values to decide PPO work. The exact generator state API and byte/device behavior remain to be confirmed statically and with later approved device tests.

## Static audit caveats / go-no-go checks

1. `nx,ny` are finite per environment instance and derive from planner geometry/resolution, not observed support. The batch allocator must bind their exact values in static schema and reject shape mismatch before work.
2. Current `score_candidates` enforces B=1, so the new training-only seam must avoid changing its inference contract. Current production model forwards already accept a leading batch dimension, but A1 sampling of the raster at node coordinates and the C2 candidate-context semantics need parity evidence later.
3. Current support digest APIs expose host hex; batch implementation must use the HIP device digest result directly and retain expected bytes in tensors. The existing helper is not suitable if it converts digest to Python bytes before update.
4. Ordinary `torch.optim.Adam.step()` increments CPU-side Python state and does not support tensor-controlled exact no-op padding. The custom state transition must be shown equivalent to existing Adam, including epsilon/bias correction placement and per-parameter step semantics. If this cannot be established from installed PyTorch/HIP source without device execution, halt for Pass 1 revision rather than guessing.
5. `torch.Generator.get_state()`/`set_state()` support and tensor residency for HIP must be established from official local API/source documentation; no generator probe or device call occurs in this tranche.

This is a design note only, not acceptance evidence. No policy parity, autograd/Adam behavior, device residency, checkpoint determinism, simulator behavior, training, or qualification is claimed.
