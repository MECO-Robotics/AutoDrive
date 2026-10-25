# REBUILT official Slice A scoring contract

## Planner scope

The scoped plan defines a pure, TU22-pinned official score-ledger contract with separate abstract proxy rows, explicit completeness gates, and null official points when required evidence is missing. The planner artifact is `docs/plans/rebuilt-match-points-ledger-20261025.md` (SHA-256 `05f11b1838643258ac57f56734e1ee33aacd511230e59dcef88942cc1713d1c9`).

## Implementer lineage

- `515a714b2d86f47cd8688465e5edf1c093114d01`: pin the TU22 rules snapshot.
- `a90369928098185983ba0ff204a8a90e53066844`: add the pure scoring contract.
- `13cebc4603b42b58538680c12d9d382e13cafc8e`: restrict penalty/disqualification/RP attribution to active lineup scope.
- `e9240c18b2c7ba2a5402461b5385056ef36ed619`: bind each TOWER award to an exact active alliance slot.
- Integration commit `f17a1c7c89da00425abc60e0b5b28ee20498e704`.
- Scoped paths and SHA-256 values at ledger base `c55141876f2d6ff2ab1102294da79c222a742219`:
  - `frc_defense/rebuilt_match_scoring.py`: `e0245096277b75e494b37d476a008afa0b0837b70da2016897dc1342fb7d3dda`
  - `qualifications/meco/rebuilt-rules-TU22.json`: `18cf92041290e8fa1df15d7e967fe5e107118a6d443fe2b6526c1c05b73f47a8`
  - `tests/test_rebuilt_match_scoring.py`: `be5ea76b19854e352ad6912792b9dd08cb3a5ca3c3d4e79952e663ab1d5bfe92`

## Independent review

The exact final feature head `e9240c18b2c7ba2a5402461b5385056ef36ed619` was approved after the TOWER attribution correction. The reviewer confirmed alliance/team/robot attribution binds to one exact active slot, cross-alliance awards are rejected, and the prior scoring completeness gates and source pins remain sound. The review evidence records contract/source inspection and the cross-alliance rejection case; no simulator, match, or device evaluation is claimed.

## Limits

This is a pure contract, not a runtime connection to complete six-robot sensor, TOWER, referee, roster, or match evidence. It produces no current official match result or qualification evidence. Missing official categories remain incomplete, not zero.
