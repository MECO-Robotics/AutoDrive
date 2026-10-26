# Independent review: P1/P2 source-authoring readiness packet

- **Verdict:** REVISE
- **Reviewed commit:** `f6138726e04287bf3684a0f222deb40cdeefff44`
- **Reviewed file:** `docs/plans/p1-p2-source-authoring-packet-20261026.md`
- **File SHA-256:** `3ce34f6b791f084d50cb3082dbbcdb444acae5fc3bf16525f2ec0e1c9a29c4b8`
- **Review mode:** static document and Git-object inspection only.

## Supported scope

The packet correctly characterizes P1.1 SOURCE/API ACCEPT as static source-map and blocker evidence only, says P1.2 fixed-byte authoring may proceed within the accepted allowlist, leaves P1.3/P1.4 blocked, retains the distinct planner/implementer/reviewer assignments, preserves the P5 trainer integration boundary, and keeps G0 as the execution/integration gate. Its accepted P1 successor plan pin and P2 corrected-plan/ACCEPT pins match the cited artifacts.

## Required corrections

1. The packet pins the original P1.1 map (`920e9b9dd500e655834c7724a1e17c7e02b81bd9`, file SHA `51ba48...`) and its review (`006804566...`, record SHA `b22c1d...`) as current. That map has been corrected at `c902659d1b9079b3afcc2e4796780d97020b6726`; corrected file SHA-256 is `c8c1d0594977ec456d8513293fe7ba0ca0020eba52a822e75cad34104aa36b4e`, Git blob `7ece927b2dfde42fb76f361d3f6438bf38a77a4f`. The corrected map states the owner-approved episode-local sequence reset under a new `episode_generation`; same-tick allocation ordering remains unresolved. The packet needs to pin this corrected exact head and its independent review when available.
2. The packet currently lists sequence “order/reset” among unresolved semantics. Reset is fixed by accepted owner response commit `3738ad58fc883c5853b32c508e89ac657864bca4`; only deterministic same-tick order/allocation remains unresolved. Revise the wording and preserve the P1.3/P1.4 stop on the actual remaining blockers.
3. The P1.2 fixed-byte ABI declaration now has an exact-head SOURCE/API ACCEPT: declaration commit `78b640630fde1aa9c0934463525e322ba57e027e`, SHA-256 `5e4c803a676e7a445b686d0f73e5d18e7341533119571eae8e9545d7a259d278`; review record commit `abc4a3d712bf829d0af5ec7a3ece9e173f7039c3`, file SHA-256 `47d1426ea8b02821a9521839d8c753a2e9e9a8d7505a42a1480fa615a1e7dff5`. The packet may retain P1.2 as permitted, but should pin this evidence and continue to state that it accepts only fixed schema declarations, not dynamic event semantics or runtime API behavior.

No tests, imports, builds, simulator, dashboard, GPU/device, campaign, or lease operations were performed.
