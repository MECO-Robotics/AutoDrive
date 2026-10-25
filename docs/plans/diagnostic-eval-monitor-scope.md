# Diagnostic evaluation monitor scope

## Planner pass

Add a manifest-bound, opt-in diagnostic monitor for a **future clean campaign**.
Do not modify, resume, or relaunch the active r4 campaign. Its evaluation
protocol and binding are null and cannot be retrofitted.

The monitor is separate from finite matched validation and final-test schedules.
It has a frozen diagnostic protocol identity and digest, exact diagnostic
seeds disjoint from the train/validation/test partitions, cycle cadence,
episode budget, a single evaluation-device assignment, and an explicit bounded
backlog policy. Each training frontier is identified by cycle/update and the
exact five immutable checkpoint snapshots plus source/config/protocol hashes.
A latest-only backlog may replace pending work only when every skipped frontier
is durably journaled; completed results bind the exact checkpoint hashes,
seed list, report/status digests, and remain permanently diagnostic and
qualification-ineligible.

Evaluation must lease only its configured device. Training on the other WX9100
continues while the monitor runs. Workers assigned to the leased device defer
without taking ownership; monitor failure, delay, or restart must not stop
training. Finite held-out schedules retain their current semantics.

## Implementer scope and review gates

- Validate monitor schema, cadence, episode budget, device, protocol digest,
  and diagnostic seed partition isolation in the manifest.
- Use a durable hash-linked cursor/journal with safe retry/adoption, immutable
  frontier snapshots, explicit skipped-frontier records, and tamper checks.
- Integrate nonblocking scheduling and device leases without modifying current
  campaign artifacts.
- Prove with mocked dispatch and journal tests that restart/retry, repeated
  cases at different frontiers, exact five-checkpoint binding, report/status
  integrity, explicit coalescing, training on the unleased device, and
  stop-file behavior are correct.
- Keep all monitor outputs diagnostic-only; never consume validation or test
  seeds and never advance qualification or promotion state.

This code tranche is not runtime evidence. After r4 releases the GPUs, a new
clean campaign must demonstrate repeated all-five monitor results while
training continues on the unleased device. That seeded run needs current
backend/frontend dashboard verification and a fresh inspected simulator replay
under the repository's `AGENTS.md` requirements. No GPU, simulator, dashboard,
or training operation is authorized by this implementation tranche.
