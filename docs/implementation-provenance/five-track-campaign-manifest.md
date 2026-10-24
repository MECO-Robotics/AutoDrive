# Five-track campaign manifest

## Planner summary

The existing five-track trainer runs on legacy pinned checkpoint and evaluation roots, so the next campaign needs a source-bound manifest before it can be safely configured. The manifest must bind all five model/config schemas, frozen protocol and split seeds, two verified WX9100s (with the P600 excluded), warm-start bytes, output namespace, and scheduler/lease contract. It must not claim Phase 2 coverage from caller-authored metadata or claim launch readiness before supervisor integration.

## Implementer summary

Commit `bc7eb5125` adds `frc_defense/meco_training_manifest.py`, an explicit-config `describe`/`validate` API and CLI, plus a host GPU inventory artifact and focused contract tests. Manifest generation is offline and does not import or start the simulator. Campaign output paths must be fresh, unique, namespaced, and separate from pinned live roots; publication is atomic and no-clobber. The inventory binds `cuda:0` and `cuda:1` to WX9100 BDFs and excludes the P600. Phase 2 target coverage, qualification eligibility, and launch readiness remain false because dataset replay evidence, trusted calibration binding, and supervisor manifest consumption are not yet implemented.

The exact implementation file hashes are:

- `frc_defense/meco_training_manifest.py`: `c9b533b0d5dd5c50775c5bbdb4898e8c93071703118484ccd8660e0b90462b94`
- `qualifications/meco/host-gpu-inventory-20261024.json`: `5f7518b9c0a25687947a5375eb01a10e67a7a9089f7264eb44625bab52b543af`
- `tests/test_meco_training_manifest.py`: `0bd06f2021e800dcd51bcbe4a80c2191094a23f293fad4f91fd86c9be2d839a0`

The focused suite passed 13 tests. Compilation, CLI help, and diff checks passed. No simulator, training, GPU, replay, or qualification run was performed.

## Reviewer summary

The independent reviewer first rejected a draft that could treat caller-authored v1 calibration metadata as qualification evidence and did not rederive the serialized source identity. The corrected exact files were approved. The reviewer verified that calibration eligibility is always false without trusted binding, source identity and hash are cross-checked, prior output namespaces are rejected, target coverage remains false, and launch readiness is hard-false. The review reran the focused suite: 13 passed.

Candidate integration commit: `50aaef203` (`Add fail-closed five-track campaign manifest`). This summary records a scoped implementation only; it does not establish warm-start evidence, qualification, safe launch, or whole-plan completion.
