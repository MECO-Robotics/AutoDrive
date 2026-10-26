# Independent pre-G0 SOURCE/API review: P1.2 score-event ABI registry

- **Verdict:** SOURCE/API ACCEPT
- **Reviewed source/ABI candidate:** `78b640630fde1aa9c0934463525e322ba57e027e`
- **Reviewed declaration:** `frc_defense/meco_score_events_v4.py`
- **Declaration Git blob:** `a79fda17b4be17a63c4bbcd237c9a777b737edf4`
- **Declaration SHA-256:** `5e4c803a676e7a445b686d0f73e5d18e7341533119571eae8e9545d7a259d278`
- **Reviewed corrected P1.1 map:** `c902659d1b9079b3afcc2e4796780d97020b6726`
- **Map path:** `docs/implementation-provenance/meco-v4-p1-score-source-map-20261026.md`
- **Map Git blob:** `7ece927b2dfde42fb76f361d3f6438bf38a77a4f`
- **Map SHA-256:** `c8c1d0594977ec456d8513293fe7ba0ca0020eba52a822e75cad34104aa36b4e`
- **Pinned source base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`
- **Accepted P1 package plan:** `40d151dec437fe405bd84d2918b2b86ae6f4a129`, file SHA-256 `1cfe7ccd56a4355501a7e46a3d3e3cd9bf0ae1c443bb0da571c27e05f6043300`; Pass 1 ACCEPT review `2810a315970c56b0161bbfbae025a7a0301ece8e`
- **Approved P0a:** candidate `56183cefd2402342ce28f836155164fc850ebc60`; maintainer record `7007607010636fe7451ad1b59eb1a900e0a4d7b1`; Pass 3 ACCEPT record `d5dab585ef1147f0d9997ece80db6e9a7ba95979`
- **Accepted owner contract:** `3738ad58fc883c5853b32c508e89ac657864bca4`; owner response review `8e58451551c82c4713688e3fd228a9b410125e2155157f437e75679b2e3601ac`
- **Review mode:** static source and Git-object inspection only; no import, test, build, or runtime execution.

## Evidence and contract checks

The ABI declaration contains only fixed scalar schema fields and constants. Field order, types, widths, offsets, and total are: `valid bool 1@0`; `rollout_row_index uint8 1@1`; `episode_event_sequence uint16 2@2`; `tick int64 8@4`; `match_elapsed_seconds float64 8@12`; `world_slot int64 8@20`; `alliance uint8 1@28`; `robot_slot int64 8@29`; `piece_slot int64 8@37`; `piece_generation int64 8@45`; `event_type uint16 2@53`; `score_units uint16 2@55`; `source_site_code uint16 2@57`. The final field ends at byte 59, with no gaps or implicit padding. Capacity is 2,304 rows and `2,304 × 59 = 135,936` bytes.

These declarations agree with the accepted V4 Gamepiece/score owner response: event type `1` is `fuel_score_awarded`; source-site codes `1` and `2` identify the vectorized and per-robot score mutation sites; each accepted mutation records one score unit; the packed record is 59 bytes; and the fixed batch capacity is 2,304. The declaration adds only the specifically allowlisted `frc_defense/meco_score_events_v4.py` file. It performs no packing, decoding, hashing, dynamic-value validation, allocation, or host numeric processing, and it adds no dependency.

The corrected P1.1 map accurately distinguishes the accepted sequence contract from unresolved source behavior: the owner response fixes `episode_event_sequence` as episode-local and says it restarts only under a new `episode_generation`; the pinned source has no sequence counter, and deterministic allocation order across same-tick vectorized and robot-loop awards remains unresolved. The ABI declaration does not make that ordering normative. It also leaves physical-generation lifecycle and tick/elapsed-time sampling unresolved, and identifies the absent canonical event-ID consumer. The map retains the P1.3/P1.4 stop and P5 integration boundary.

## Acceptance scope and limits

This SOURCE/API ACCEPT covers the fixed P1.2 ABI declaration and the corrected P1.1 evidence/blocker map at the exact pins above. It does not accept an event producer or consumer API, establish dynamic sequence/allocation behavior, close physical-generation or tick/time semantics, or authorize P1.3/P1.4. Those remain blocked pending the reviewed owner/source decisions and gates in the accepted P1 plan. Any change to either reviewed artifact, source base, or contract pins requires a new exact-head review.

No tests, imports, builds, scoring calculations, simulator/replay, dashboard, calibration, GPU/device, training, integration, campaign, or lease operation was performed. Runtime behavior and GPU residency remain unvalidated.
