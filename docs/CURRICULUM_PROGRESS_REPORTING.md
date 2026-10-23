# Curriculum performance reporting

The unattended queue writes `orchestration.json` and each training lane writes
`stats.json` and `evaluation-N.json`. Generate an isolated, simulator-free report:

```bash
python3 scripts/report_curriculum_progress.py \
  --root evaluations/curriculum-variants-20261021 \
  --output evaluations/curriculum-variants-20261021/report/progress.json
```

This atomically publishes `progress.json`, `progress.md`, and `performance.html`.
The HTML refreshes every 30 seconds; the queue or a service timer must regenerate
the files. Serve only the report directory so checkpoints and raw artifacts are
not exposed by the report server.

Every saved evaluation compares the learner to the deterministic observed teacher
on the same scenario, category, seeds and evaluation protocol. This teacher uses
generic NN execution; it is distinct from the deployed native deterministic
controller. Teacher qualification is measured separately for each category.
An unqualified baseline can support a descriptive ratio but cannot satisfy a
qualified-reference curriculum gate.

Progress tables include queued lanes before their first completed update,
historically qualified scenarios, current practice scenarios and consecutive
gate passes. Saved statistics alone do not establish worker liveness; the queue
manifest supplies process status. A manifest older than three minutes is marked
stale in the HTML. A running update, a pending evaluation and a failed lane are
different states and should be reported separately.

The JSON `comparison_group` identifies matching scenario, category, seed set,
split, source identity and protocol. Do not combine unmatched groups into an
architecture ranking. Training update counts remain separate: equal evaluation
conditions do not imply equal training budgets. Each training seed is a replicate;
episodes from a single seed do not count as independent training runs.

Success rate, its saved 95% Wilson interval, useful throughput, deterministic
success and reference qualification are reported separately. Training reward
is not a substitute for physical success. Partial or incompatible evidence is
listed under errors and excluded from comparisons. Diagnostic evaluations stay
explicitly marked diagnostic.

The reporting tests operate on saved JSON fixtures and do not simulate robot
behavior. End-to-end curriculum and dashboard validation belongs to the launch
qualification record, rather than being inferred from these parser tests.
