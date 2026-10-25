# Plan A human choice authority v1

`frc_defense/meco_plan_a_expert_choice.py` introduces a standalone contract for
human-selected feasible Plan A node/behavior pairs. Ed25519 signatures bind the
annotator identity and basis to sidecar, source, and replay content hashes. The
validator joins exact decision identities across the source, replay-validated
legal support, and actual-execution evidence; it rejects duplicate or
incomplete coverage, non-train use, infeasible/remasked choices, state/support
mismatches, and failed or preempted execution. Pose data comes from the
validated canonical support row. The existing minimum-route producer remains a
baseline and is not labeled human expert.

Focused CPU tests use synthetic fixtures and generated signing keys. No real
source collection, simulator execution/replay, GPU operation, dashboard check,
or warm-start fit was performed. Production fitter integration and a pinned
deployment key registry remain required before this can produce trainable
artifacts.
