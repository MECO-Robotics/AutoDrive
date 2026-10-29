# Runtime repair validation — 2026-10-17

Scope: opt-in simulation utilities in `counter_motion.py` and
`runtime_evidence.py`. The default controller and collection1668/RETURNc70
weights are unchanged. No physical hardware or dashboard was used.

## Focused tests

```sh
python -m pytest tests/test_counter_motion.py tests/test_runtime_evidence.py -q
```

35 tests passed. The source milestone is commit
`2299ee621921143b9944e61b7f4cdaed55555bdd`. These tests cover source-occurrence
selection, default/configuration isolation, surface conventions, actual-actor
attribution, hash/call reconciliation, bounded buffering, latched overflow and
explicit write failures. Tests alone do not establish simulator behavior.

## Coupled simulator evidence

A separate preregistered harness exercised the maintained utilities directly.
It retained six fixed USED tasks, with one worker at a time and no training:

- Four FUEL subsystem trials: two saved drum-contact states × signed factors
  +1/−1, 100 real 1ms steps each. New configuration code built the actual
  backend metadata. Every trial passed independent contact/force checks,
  movement, ownership/projection consistency and registered query/impulse
  caps. These use declared synthetic zero clocks and a stationary reset
  chassis; they are not historical whole-world restorations or collection tests.
- Two seed 94000 C60 collect-to-return trials: collection1668 with RETURNc70,
  and collection1668 with feedback RETURN. Both retained all 60 physical
  pickups, arrived with 25-tick dwell, and matched their archived same-arm
  non-timing command/state/event traces. Both had zero prohibited contacts,
  invalid/domain/projection violations and runtime AD* calls.

| C60 observation | Neural RETURN | Feedback RETURN |
|---|---:|---:|
| Completed physics ticks | 1248 | 1255 |
| First60 pickups, simulated seconds | 20.70 | 20.70 |
| Final arrival, simulated seconds | 24.96 | 25.10 |
| RETURN decisions / actor forwards | 43 / 43 | 44 / 0 |
| Actual RETURN actor | c70 | none |
| Decision-entry-to-step p95 / max, ms | 2.140 / 8.236 | 2.154 / 11.073 |
| All-mode decision p95 / max, ms | 1.986 / 8.095 | 1.966 / 10.954 |
| RETURN forward-through-decode p95 / max, ms | 0.255 / 0.337 | N/A |
| Deferred durable evidence drain, seconds | 0.999 | 0.607 |

All-mode decision timing ends when its record is assembled. Entry-to-step also
includes row serialization and pre-step capture; neither includes the physical
step or final drain.

Pure neural RETURN forward p95/max was 0.103/0.126 ms. Timer spans overlap and
must not be added. Observation, serialization and handoff snapshot work remain
in the measured synchronous path; file/directory fsync and historical-reference
reads do not occur there. Startup/preload and final drain costs are not free.
The existing serial gates (entry-to-step max 100 ms; neural RETURN command
p95 ≤ 10 ms and max ≤ 100 ms) passed without threshold changes.

The historical 9.1667 s serving sample included synchronous handoff evidence I/O
and reference reading; its raw forward was 0.214 ms. Missing per-operation
historical timestamps prevent apportioning that delay or proving storage was
the sole cause. These serial measurements do not establish worst-case behavior
under contention or hardware realtime qualification.

## Receipt correction and retained failure

All six physical workers, coordinator and launcher have actual exit0 receipts.
The original passive aggregate exited1 because it required the post-warm
228-library map in pre-warm start receipts. That failed result remains immutable.
A separately registered, reviewed POSTHOC passive checker examined ALL six
existing trials with zero new physics, forwards or optimizer updates:

- Only the coordinator start and six outer starts use their exact recorded
  87-library pre-warm map, zero activity, and matching source/input pins.
- Full 228-map equality remains required at post-warm physical starts/ends,
  outer ends and coordinator end. Current binary rehashing is labeled post-hoc
  byte integrity, not evidence of unrecorded historical loading.
- Source/model/state, returned artifacts, unique pickups, physical/RNG anchors,
  individual actual exits, errors and numerical gates remain checked. Opaque
  tensor hashes were not independently reconstructed.

Three independent reviewers passed source/preregistration, prelaunch, and
terminal/code review. The passive checker and its parent exited0. Its result
supports bounded USED-case code validation, not a rewritten primary result or
fresh policy qualification. The retained artifact namespace is
`outputs/collection-runtime-fixes-20261017` in the experimental checkout;
large raw evidence and model files are deliberately excluded from this patch.

| Artifact | SHA256 |
|---|---|
| Six-task registration | a2ac44953c22207c7fc7d7fc00e2e38d5134554dcc696013d2ba755f0c6179b6 |
| Original failed aggregate | 73390f21559b4eaa9a01e2520fc573aa0b24645cbc01bf6daf6615c9e15d3c9f |
| Post-hoc validation result | e80a4c3c02efd352acbfb3ac801971a8e6622427aabd53e7f5be8ec1ce09964c |
| Post-hoc root receipt | 50863e0905a922b6fda1b43793f85d6f7b8b3cc61885ced3a544bee192c209a0 |

## Mechanics and adoption limits

The separate seed 95016 nominal drum-sign contrast completed two 500-tick
counter-flow trials. Control exactly matched its archived 500 rows/10,000 inner
steps and produced no sensor counts/exits. The nominal functional convention
produced four sensor counts and exits, with zero prohibited contacts. Its first
changed surface velocity/impulse was at 6.673 s, after 6,672 matching inner steps.
That result is `c51d7981fdbdf497df6ae112a709c1643051279a6910e63c3bd07a0ef8ef696f`.

This is a source-grounded nominal convention with a static-bolt assumption,
not measured motor polarity/RPM or as-built clearance validation. Shaft motion,
whole-counter fidelity, official scoring, full-match performance, model
promotion, production/load latency qualification and deployment remain
unresolved. The helpers generate no navigation and introduce no default
controller behavior. No dashboard replay was generated or reviewed, and no
current-dashboard-stack claim is made. This patch may be reviewed in a PR;
it does not authorize merging or hardware operation.
