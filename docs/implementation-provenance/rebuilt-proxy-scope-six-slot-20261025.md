# Six-slot simulated proxy scope correction

## Planner scope

The scoped Slice B plan identified that the existing abstract FUEL proxy was attached to the six-slot MECO simulator but labeled `two_robot_2d`. It required a precise simulation-only scope label while preserving the proxy/official-score separation and every non-qualification flag. Planner source: `docs/plans/rebuilt-active-hub-slice-b-implementation-20261025.md`, SHA-256 `51c7e0342d276fef1f8d70abf11e88b6d079212742d46ad94e494a7432b38815`.

## Implementer lineage

- Feature commit `01543a6f81451c182cd8834ca67b17b52800f1f2`: change the scope label to `six_slot_2d_simulation`.
- Integration commit `9d89720e3b99ab08074402e60b8c65433546c7b7`.
- Scoped paths and SHA-256 values at ledger base `c55141876f2d6ff2ab1102294da79c222a742219`:
  - `frc_defense/rebuilt_match_scoring.py`: `e0245096277b75e494b37d476a008afa0b0837b70da2016897dc1342fb7d3dda`
  - `tests/test_rebuilt_match_scoring.py`: `be5ea76b19854e352ad6912792b9dd08cb3a5ca3c3d4e79952e663ab1d5bfe92`

## Independent review

The exact feature head `01543a6f81451c182cd8834ca67b17b52800f1f2` was approved. The reviewer verified that only the proxy scope label changes, tests guard the six-slot simulation wording and preserve proxy/official separation and non-qualification flags, and no official team or roster identity is asserted.

## Limits

The six slots are simulator slots, not evidence of six official active robots. The proxy remains `simulation_only=true` and `qualification_eligible=false`; the correction supplies no sensor-verified scoring, complete official match context, official points, or qualification evidence. Slice B's full behavior implementation and end-to-end simulator evidence remain separate work.
