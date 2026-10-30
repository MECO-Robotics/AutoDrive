# Speedup audit and next pass (2026-10-30)

## Changes selected for main

Two default-off simulator fusions are combined behind independent environment gates:

- `AUTODRIVE_EXPERIMENTAL_CONTACT_ACCOUNTING_FUSION=1` clears per-tick contact flags in the fused swerve/pose kernel and fuses safety counter updates into the two-robot contact pipeline where available.
- `AUTODRIVE_EXPERIMENTAL_FUSED_SIM_TICK_ACCOUNTING=1` fuses six-robot safety counters, physics substep index, and physics time increments in one HIP launch.

The combined A/B/A run on WX 9100 HIP 7.17 used production A1 `full_match_offense/active/open/level_5_mixed`, seed `2970053994459693056`, one world, six robots, six paired ticks after warmup. Median was 24.505 ms/tick for baseline and 20.744 ms/tick with both gates enabled: 1.1813x. A0/A1 baseline repeat ratio was 0.9980; persistent simulator tensors matched exactly after every paired tick. This remains a gated optimization, not A1/A2/A3 qualification.

## Rejected or unproven candidates

| Candidate | Measurement | Decision |
|---|---:|---|
| Trailing route padding prune | 1.00038x over 26 captures; exact on every capture | Reject; no material gain |
| Shared robot geometry precompute | 1.00512x, exact per macro | Reject; noise-sized gain |
| Robot wall fusion | 0.93440x, exact per macro | Reject; slower |
| Deferred static-contact `any` | 0.96654x, exact per macro | Reject; slower |
| FUEL free-mask hoist | 0.76962x and parity failure in an unsuitable `counter_defense` harness | Invalid production evidence; do not promote |

## Simulator next pass

The combined counter and flag work yields 1.181x on the isolated six-robot physics tick. It does not meet the overall 2x objective. The retained A/B/A artifact measures both gates together, not either gate independently. The reported whole-macro profile has 6,303 HIP kernel events, 29.391 ms summed device-kernel time against 45.530 ms synchronized macro wall time; generic elementwise kernels account for about 14.294 ms across 3,927 events. That attribution is from a single run, and its raw profiler trace was not retained, so it is not independently inspectable or paired. Do not use those counts to justify a specific kernel change.

Next candidate to investigate: command preparation in `TensorVectorizedSimulator._step_eager` (NaN handling, omega clamp, translation norm/clamp, speed desaturation, and concatenation), potentially fused into the existing six-robot swerve/pose HIP entrypoint. First capture paired A0/B/A1 synchronized profiler traces on the same warmed, seeded production A1 macro. Retain raw traces, call stacks, source/native identities, and counts plus summed/median time for those operators and the fused entrypoint. Only prototype if this sequence is a material measured cost. Isolate the new gate from both existing gates, preserve inactive-mask and NaN/Inf behavior and arithmetic order, and require exact persistent-state hashes after every macro in paired A/B/A timing. Reject if parity fails, speedup is within noise, or A1/A2/A3 qualification regresses. Any behavior-changing candidate also needs the representative simulator and current-dashboard replay validation required by `AGENTS.md`.

## Route next pass

The A1 route profile records 16.35 ms p50 for `build_plan_a_support`, 2.41 ms p50 for support-tree digest, about 125 ms p50 for route-support wall time, and 371.5 ms p50 / 18.47 s p95 for the broad route pipeline (five samples with extreme outliers). Trailing-padding pruning matched outputs but measured 1.00038x, so that path is not the next target.

The normal HIP graph route runs a 1024-thread ordered-frontier relaxation and scans node keys each frontier round. It stable-sorts route costs afterward. A queue-based replacement could reduce scans, but must preserve node-ID ordering for equal routes, the strict 1e-9 relaxation threshold, first-parent tie behavior, zero-cost edge handling, and serial fallbacks. The saved `.pt` captures are support outputs, not the graph and root tensors passed to `dijkstra_graph`; no suitable low-level solver inputs were found. Next step is therefore to capture production A1 graph/root inputs with scenario, seed, checkpoint, and source provenance, then add instrumentation-only counters and per-stage timings. Record rounds, `n × rounds` node-key visits, frontier sizes, relax proposals/successes, incoming comparisons, zero-boundary rounds, fallback reason/frequency, and timings for graph preparation, frontier solve, stable sort, and total. Only prototype an ordered active frontier if scan counts dominate measured time; compare route tensors, status, parent edge lengths, traversal order, support masks/digests, and selected actions exactly.

See [route-hotpath-research-20261030.md](route-hotpath-research-20261030.md) and [route-frontier-scan-instrumentation-plan-20261030.md](route-frontier-scan-instrumentation-plan-20261030.md) for source-level contracts and capture details.
