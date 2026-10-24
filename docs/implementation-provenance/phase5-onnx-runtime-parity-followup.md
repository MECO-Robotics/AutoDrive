# Scoped Three-Pass Summary #34: Phase 5 ONNX Runtime parity follow-up

## Planner summary

The existing parity verifier already created an ONNX Runtime session, but the
path had no graph-execution contract covering the five adapters and its
production `ExportAdapter` tuple outputs did not match the verifier's output
handling. The scoped fix must retain canonical input names/order, execute the
exported ONNX graphs, and distinguish synthetic CPU evidence from real capture,
trained-checkpoint, Hailo, and device qualification.

## Implementer summary

Reviewed implementation commit:
`8cc96a9270930ca86147b00efa60404e062bbe6e`.
Integrated candidate commit: `7c932da57573b6fd0b0adaaf7b2d01083f275d1b`.

The change fixes the recurrent input/output name collision by keeping the input
named `hidden` and naming the ONNX output `hidden_out`. Seeded A2 node headings
are now valid integer bins. The verifier handles both direct model output
mappings and the production adapter's ordered tuple. New CPU synthetic tests
export and execute actual ONNX graphs for A1, A2, A3, C1, and C2, check numeric
parity and digests, cover both model-output forms, and reject input-name drift
and out-of-tolerance outputs.

Changed paths:

- `frc_defense/meco_deployment.py`
- `scripts/verify_meco_onnx_parity.py`
- `tests/test_meco_deployment_contracts.py`
- `tests/test_meco_onnx_runtime_parity.py`
- `tests/test_meco_parameter_inventory.py`
- `docs/implementation-provenance/production-input-onnx-parity.md`

Focused validation passed 20 tests across ONNX runtime parity, parameter
inventory, and deployment contracts. It used CPU `CPUExecutionProvider`, ONNX
1.19.1, and ONNX Runtime 1.23.2 in an isolated worktree-local dependency
environment; that temporary environment was removed after the run. Python
compilation and `git diff --check` passed.

## Independent reviewer summary

The exact implementation commit `8cc96a9270930ca86147b00efa60404e062bbe6e`
was independently reviewed and approved before integration at
`7c932da57573b6fd0b0adaaf7b2d01083f275d1b`.

## Limits

The verifier implementation predated this tranche; this pass fixes its
synthetic input/output handling and adds real ONNX Runtime graph-execution
coverage. Evidence uses synthetic inputs and random initialized weights. No
trained checkpoint or real production capture was tested. No Hailo provider,
compiler, device, robot hardware, simulator, or deployment qualification was
tested. The production capture/export-manifest binding path remains unrun.
