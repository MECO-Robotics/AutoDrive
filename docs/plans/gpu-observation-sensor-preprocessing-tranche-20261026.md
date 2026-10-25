# Pass 1 plan: GPU-resident observation and sensor preprocessing

**State:** implementation plan only; static source review, no code/runtime changes.  
**Reviewed base:** `53f3d9c9a49a7e12bf65b5e1fa2797f4464e63be`.  
**Execution constraint:** the live diagnostic r4 campaign remains untouched and uninterrupted. Defer fixtures, GPU/device checks, profiling, and simulator evaluation until r4 is manually stopped and both WX9100 leases are released.

## Objective and boundary

Make the simulator observation and sensor-preprocessing data that feeds policy inputs GPU resident from active-world selection through perception/tracking, observation assembly, normalization, and the returned observation tensor. Device masks and tensors must determine all per-world/per-robot work. No CPU read of a tensor predicate may branch, compact, choose, or alter policy-visible values.

This tranche covers `_tensor_3v3_observation` active-world sampling/branching, the `observation.py` NumPy normalization adapter, and `tensor_sim.py`/`tensor_defense_observation.py`/`tensor_3v3.py` perception and observation callsites feeding policies. Candidate generation/scoring and PPO optimization/trainer-loop changes are separate work and are excluded. Static configuration such as robot roles, fixed capacities, field dimensions, and a caller-supplied count used only to size a known batch may remain host metadata; values read from runtime device masks may not drive host control flow.

## Source callchains in scope

1. **3v3 policy observation:** `tensor_3v3.py::reset/step` → `_tensor_3v3_observation.py::_update_perception` → sensor detections, anonymous tracking, and opponent tracking → `observe(active_mask)` → `[world, robot, 137]` observation tensor. This is the primary six-robot policy observation path.
2. **Defense/strategic observation:** `tensor_defense_step.py` and `tensor_defense_reset.py` → `tensor_defense_observation.py::_update_perception`, `_raw_obs`, `_strategic_observation`, `_obs` → policy observation history and current state. Maintain its active mask and cache/history merge semantics.
3. **Normalization:** `observation.py::normalize_numpy_observation_in_place` is a separate NumPy path; `normalize_tensor_observation_batch_in_place` is the torch path used by both observation mixins. Policy-facing normalization must use a device-resident tensor implementation with the same fixed scales and clamps.
4. **Sensor kernels and observation prototypes:** `tensor_perception.py`, `tensor_strategic_observation_proto.py`, and `tensor_strategic_observation_full_proto.py` are called from the above paths. Keep their observation-visible intermediate tensors on the selected device; any optional extension/prototype selection must be static capability/configuration, not a data-dependent host predicate. If an extension cannot satisfy the device contract, fail closed for the GPU-only qualifying path rather than silently switching to a host observation implementation.
5. **Policy handoff:** `env.observe()` and `_obs()` return device tensors consumed as `_last_obs` by policy decision code. This tranche ends at that tensor boundary. Action selection, candidate construction/scoring, and PPO update/rollout code are outside scope.

## Current issues to resolve

- `_tensor_3v3_observation.py::_random_active` computes `int(active.sum().item())` when `active_count` is absent, then uses that host count to allocate a compact random tensor. This creates a device synchronization and a data-dependent host branch. The active mask must remain a device tensor; random draws and masked writes need a GPU-only design.
- `_obs` and 3v3 `observe` include Python branches such as partial-batch checks and first-frame logic. These must be based on static caller metadata where available or represented as tensor masks and tensor selects. Do not call `bool(active.all())` or otherwise materialize a device predicate to decide a host branch.
- `tensor_defense_observation.py::_random_active` has the same `.sum().item()` fallback. Existing callers sometimes pass `active_count`; preserve that count only when it is known outside the device tensor and is used solely for allocation/shape. Do not derive it by reading the mask on CPU.
- `observation.py::normalize_numpy_observation_in_place` has host scalar/list arithmetic, and may be used by NumPy adapters. Keep it for explicit offline/CPU compatibility if necessary, but exclude it from qualifying GPU policy paths. GPU validation must prove the active policy callchain uses only the tensor normalizer.
- The observation/perception code contains static Python branches over configured control modes/roles and feature availability, plus host flags returned by optional kernels (`fused_tracks`, `fused_opponents`). Classify each: static configuration/capability branching can remain; runtime tensor outcomes must be applied through device masks and cannot decide host execution or skip required work.
- Existing paths use separate CPU-specialized visibility logic and optional fused kernels. The GPU-only route must not use CPU-specialized paths or fallback based on a runtime tensor condition. Device dispatch may be selected from the explicit configured device/capability before sensor values are processed.

## Device-resident shape and control contract

- Active worlds: boolean tensor `[N]` on `env.device`; every update is masked with this tensor. No `active.any/all/sum`, tensor-to-bool conversion, `.item()`, `.tolist()`, `.cpu()`, or NumPy conversion may control a host branch in the qualifying path.
- 3v3 observation output: floating tensor `[N, 6, 137]` on `env.device`; `OBS_DIM=137`. Row updates and preservation of inactive worlds use tensor `where`/masked writes.
- Sensor visibility and anonymous detections: device masks such as `[N, 6, F]` or configured camera-expanded forms; positions/velocities `[N, 6, F, 2]` (or their documented camera/observer-expanded tensors), with validity, confidence, extents, and merge markers aligned on the same device.
- Track state: bounded position/velocity `[N, 6, F, 2]`, age/confidence/validity `[N, 6, F]`; opponent estimates `[N, 6, ...]`. Association, one-to-one collision resolution, stale-track expiration, and active-world commits must be tensor operations on `env.device`.
- Defense raw observations: fixed schema batches `[N, 35]` or the configured strategic width, normalized in place by the tensor normalizer. History writes/merges use device indices/masks and preserve inactive rows.
- Configuration values (e.g. fixed robot count, static mode tuple, field extents, model capacity) may be host scalars. A host scalar count may be accepted only as explicitly supplied shape metadata from a caller that knows its batch construction; it must not be computed from a device mask. The no-count path must remain correct without a host predicate, using fixed-shape draws and device masking or an equivalent device-only strategy.
- Preserve sensor RNG determinism as an explicit contract. If fixed-shape device draws change random consumption relative to compact active-only draws, version the observation/runtime identity and regenerate compatible seeded fixtures after device release; do not silently claim seed parity.

## Implementation sequence (future approved tranche)

1. Document and classify every active-mask read, branch, RNG call, perception kernel dispatch, normalization call, and output merge along the callchains above. Mark each host operation as static configuration, shape metadata, passive logging, or prohibited decision predicate.
2. Replace mask-derived host counts and boolean branches with device-resident masks and fixed-shape tensor operations. Reconcile RNG draw shape/order with current semantics and version identity if exact parity cannot be retained.
3. Ensure every perception result that feeds policy state—visibility, detection acceptance, track association, confidence, opponent validity, and stale-state updates—stays on the configured GPU. Gate optional kernels using static capability and assert their output device/shape; qualifying mode must error on unsupported/mismatched paths.
4. Enforce the tensor normalization path on GPU policy callchains. Keep the NumPy adapter only for explicitly non-qualifying CPU/offline consumers; add a fail-closed guard at the policy observation boundary if a host/NumPy result is presented for a GPU run.
5. Add device and shape assertions at perception entry/exit, normalization, history merge, and observation return. Ensure no partial-batch or first-observation host predicate is read from device; fail closed when required initial state is absent.
6. After code review, wait for manual r4 stop and release of both WX9100 leases. Then run deterministic GPU fixtures for full/partial active batches, first observation, empty detections, stale tracks, configured sensor modes, and normalization. Verify output values and RNG contract against approved reference fixtures.
7. Run a seeded end-to-end simulator evaluation that exercises the changed sensor/observation behavior and inspect its record/replay. Restart dashboard services from the reviewed checkout, then verify service health, backend API, and frontend asset identity before creating/reviewing a fresh replay, as required by `AGENTS.md`.
8. Freeze exact source identity only after all code changes. Recompute protocol/calibration bindings and verify checkpoint compatibility against the exact observation schema, normalization, sensor RNG semantics, runtime and source hashes. Do not reuse a prior qualification/calibration claim after observation semantics or random-stream behavior changes.

No code, tests, simulator, dashboard, device lease, GPU command, or campaign modification is authorized in this planning tranche.

## Acceptance evidence for later implementation

- Static callchain review shows the policy observation path contains no runtime device predicate read on CPU and no NumPy observation normalization on GPU-qualified execution.
- Instrumented runtime confirms active masks, perception outputs, tracking state, normalized observations, and history buffers stay on the configured GPU, with fixed documented shapes and correct inactive-row preservation.
- Deterministic GPU fixtures cover full and partial active masks, static control-mode combinations, detection/no-detection, track association/expiry, first-frame behavior, and every normalized feature scale. They establish unchanged values/RNG behavior or explicitly versioned new behavior.
- Negative cases (wrong device, missing kernel/capability, malformed shape, unavailable initial observation) fail closed; they never select a CPU observation fallback.
- A seeded end-to-end simulator evaluation records scenario, seed, source hash, and outcome; a fresh replay is generated/reviewed only after dashboard backend and frontend currentness are verified.
- Evidence identifies the exact source/runtime and immutable qualification/calibration binding and confirms exclusive use of both WX9100 leases after r4 release. Diagnostic r4 output is not qualification evidence for the changed source.
