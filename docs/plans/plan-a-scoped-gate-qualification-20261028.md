# Plan A scoped HIP gate qualification

The graph-geometry entry used by Plan A training and the legacy preweighted entry have separate source constants, both false by default. Qualification of one does not qualify the other. The candidate harness calls the underlying extension for evidence while requiring both production constants to remain false.

## Candidate evidence

Commit the closed-gate checkout and verify it is clean. The fixture manifest pins the candidate sources and fixture payload; preserve the manifest and the exact checkout revision with every report. From this checkout, with the project ROCm Python environment, run:

```sh
PYTHONPATH=. python scripts/verify_plan_a_hip_parity.py --mode offline --report-dir "$EVIDENCE_DIR/offline"
PYTHONPATH=. python scripts/verify_plan_a_hip_parity.py --mode device --device cuda:0 --warmups 3 --repeats 10 --report-dir "$EVIDENCE_DIR/cuda0"
PYTHONPATH=. python scripts/verify_plan_a_hip_parity.py --mode device --device cuda:1 --warmups 3 --repeats 10 --report-dir "$EVIDENCE_DIR/cuda1"
```

Coordinate device use with other work; the harness acquires its exclusive device lease and checks physical device identity after leasing. Record each report, console output, checkout revision and source/build identity. Investigate failures rather than weakening fixtures, checks, leases or routing. Assess graph geometry parity, invalid inputs, support and route outputs, latency distributions, memory, compile and first-use overhead on both devices. Preweighted performance and parity must be assessed separately if its gate is ever considered.

## Acceptance and source binding

A passing candidate report is evidence, not an authorization to change a gate. Review the current project acceptance plans, including the full device-resident decision slice, production dispatch, failure behavior, representative seeded simulator replays, current dashboard provenance and end-to-end training throughput. Candidate extension parity alone does not establish those contracts. Independently review the evidence and the proposed scope before any enablement.

Enabling a constant changes a manifest-pinned source file. Candidate reports are bound to the closed-gate revision and the harness refuses to run with either gate enabled. Therefore an enablement proposal must explicitly bind the candidate revision to the final enablement delta, update source attestation as appropriate, and separately validate the enabled production dispatch and the still-closed preweighted path on the final source. Do not present closed-revision reports as evidence generated from the enabled revision. The current task does not enable either gate.
