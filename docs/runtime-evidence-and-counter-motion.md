# Simulation evidence and counter motion

These opt-in utilities repair experimental simulation/evidence paths. They do
not change the default robot controller, train or adopt a policy, calibrate real
motors, or establish official scoring fidelity or hardware realtime operation.
Frozen historical experiment sources and results remain unchanged.

## Nominal drum coordinates

`frc_defense.counter_motion.configure_drum_motion` selects the drum by its complete
CAD occurrence path and expected part name. It returns a private manifest copy;
all fields except that occurrence's signed `omega_per_surface_speed` remain
unchanged. The default factor `+1` preserves the existing model. Explicit factor
`-1` represents the separately reviewed nominal functional convention. No
compiled triangle or part index chooses the occurrence.

The source axis and contact point must share the manifest's world coordinate
frame. Surface velocity is `cross(axis * signed_omega * ramp, point - origin)`;
the existing ramp reaches full command at 0.25 seconds. Neither the sign nor
unit command establishes an actual motor polarity or measured RPM. Static bolt
assumptions, shaft motion, as-built surfaces and full counter fidelity remain
separate questions.

## RETURN attribution

`return_attribution(return_actor)` derives both mode and the boolean
`navigation_input_to_NN` from the actual RETURN actor reference. A feedback arm
has no RETURN actor; it must not inherit the collection actor's attribution.
`TrialEvidence.return_physics` emits that metadata on a copy of a physical
record without changing its commands or state.

The independent frame check is strict about boolean types. The run check also
requires actual, unchanged actor file/state hashes and forward counts matching
served RETURN decisions, or null hashes and zero forwards for feedback. A
caller must independently derive the decision count from delivered trace rows,
obtain hashes from the actual loaded actor, and check expected checkpoint pins.
Neither a presence flag nor an arm name substitutes for those receipts.

## Keep archival writes outside control

1. Before control starts, verify and preload any comparison reference with
   `load_bound_reference`. Persist actual start/source/native/actor/state/RNG
   receipts separately before steps or forwards.
2. Construct `TrialEvidence` with an explicit serialized-byte limit. Trace
   serialization and handoff capture/compare remain synchronous memory work.
   `trace()` accepts complete newline-terminated text; an independent audit
   must parse and validate every emitted JSON record.
3. At the actual handoff tick, call `handoff(actual)` with the same snapshot and
   RNG facts. This compares the preloaded reference in memory. Named auxiliary
   JSON records can be queued with `archive_record`.
4. After the physical loop, drain before writing physical-end and before audit.
   Context exit also drains on exceptions. Files and directory are fsynced;
   `drain_seconds` includes that durable I/O. Failed writes retain partial files
   and `drain_error`; there is no silent overwrite or retry.
5. Record `invalid_error`, `drain_error`, `written` and completion status in the
   outer failure receipt. Overflow stays invalid even when caught, drains the
   accepted prefix and raises. No trace record is silently dropped. Disk
   failure can also prevent a failure receipt; report it as unavailable.

`complete` describes evidence-buffer status only. It does not establish task
success, complete worker receipts, source integrity or a passing physical audit.
The serialized-byte limit excludes Python object overhead. Bound trial length
as well as bytes; these utilities are not a general-purpose logging service.

Measure NN forward, forward-through-decode, decision-entry through physical-step
entry, complete physical step, and deferred drain separately. Preload/startup
cost is not free. Nested spans are not additive. Moving archival I/O does not
prove the cause of a historical tail or guarantee latency under contention.

## Validation and qualification

Focused tests: `python -m pytest tests/test_counter_motion.py tests/test_runtime_evidence.py -q`.
They cover defaults, source identity, nominal tangents/ramp, invalid parameters,
actor attribution/hash/count mismatches, bounded ordering, caught overflow,
exception draining and explicit write/fsync errors. Unit tests alone do not
establish simulator behavior. Seeded coupled source checks, actual process
receipts and their independent reviews are required before the code-validation
report can pass.

No change here clears the retained model/serving adoption hold. Broader serving
latency needs a separately frozen load qualification; full-cycle and official
readiness need their own physical/geometry and accounting evidence. No dashboard
stack or physical hardware is started by these modules.
