# Phase 2 matched C1/C2 campaign descriptor and executor

The scoped implementation adds a metadata-only descriptor and validator for a
matched C1/C2 Phase 2 run. The descriptor binds a clean source commit and
source-tree digest, the frozen qualification protocol path/ID/file digest,
unique train-only seeds, an explicit indexed device, fixed C1/C2 collection
settings, and a new isolated output root. Description and validation do not
write campaign outputs or acquire a device lease. Descriptor file writing is
separate and explicit.

The explicit executor acquires the shared device lease, claims the fresh output
root, revalidates source and protocol identity, collects one shared matched
prefix dataset, then fits and validates C1 followed by C2 on that same dataset
and device. The report records stage order, source/protocol bindings, dataset
and artifact hashes, record and pair counts, updates, and failure status.
Failure reports retain `phase2_ready: false` and
`qualification_eligible: false`; successful artifact validation also leaves
both false pending independent evidence review and representative simulation.
This tranche does not implement Plan A, create data or weights, or qualify a
policy.

Independent review approved the exact corrected implementation after the
documented seed examples were changed to values in the frozen train split.
Focused CPU-only tests cover descriptor source/protocol/seed/path validation,
symlink and overlap rejection, no-write description, explicit writing,
ordered mocked collection/fitting/validation, failure reporting, and lease
cleanup. The 12 focused tests passed; Python compilation and `git diff --check`
passed. Review/testing performed no collection, fitting, simulator, replay,
training, device probe, or GPU work.
