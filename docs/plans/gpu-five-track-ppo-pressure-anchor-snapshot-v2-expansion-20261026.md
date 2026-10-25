# Pass 1 superseding plan: all-five PPO, reset residency, and versioned pressure-anchor provenance

**Status:** proposed scope expansion after Pass 2 stopped before edits; no implementation or runtime evidence.
**Planning base:** origin/codex/gpu-five-track-ppo-pressure-anchor-expansion-plan-20261026 at d5bb8856399edf9b8119a0a93e2963a30ec1f459.
**Supersedes in full:** docs/plans/gpu-five-track-ppo-pressure-anchor-reset-expansion-20261026.md, SHA-256 c5c5feafb808e495ac074f6f7782d6ee39c39a0024d0ff59518593b0a51ed2af, and all focal/profile/reset, rollout/candidate and PPO predecessors it incorporates.
**Requirements:** docs/plans/whole-compute-gpu-only-meco-plan-20261025.md, steps 4–8; docs/MECO_AutoDrive_Learning_Tracks_Evaluation_Plan.md, §2, §3, §5 Phase 3; repository AGENTS.md simulator/dashboard validation rules.

## New Pass 2 stop evidence

The pressure-anchor expansion requires a new resolved scenario/report schema, but the schema owner was outside its authorized scope. frc_defense/meco_baseline_freeze.py defines EFFECTIVE_SNAPSHOT_SCHEMA as v1, EFFECTIVE_FREEZE_SCHEMA as v3, the exact snapshot top-level key set, snapshot validator, receipt compiler/validator and baseline-freeze validator. Its current v1 snapshot binds static pressure configuration under resolved_fields.opponent_controllers and the opponent_schedule artifact. It has no declared per-world pressure-anchor state contract.

Relevant provenance flow:

1. curriculum_scenarios.py::_effective_curriculum_snapshot builds resolved pressure and opponent-controller values, configuration artifact bytes, producer identity, snapshot hash and schema.
2. meco_baseline_freeze.py::_effective_snapshot_body requires an exact top-level schema and verifies its version and digest. compile_effective_capture_receipt regenerates the snapshot from the public factory arguments, captures authoritative file hashes, verifies the resolved field closure and config artifact bytes, then seals a receipt.
3. validate_effective_capture_receipt requires the current source identity and exact source hashes, reruns the current factory snapshot and rejects any difference. It verifies field/artifact closures and receipt digest.
4. compile_effective_baseline_freeze emits a v3 freeze from the validated receipt; validate_effective_baseline_freeze re-compiles and requires exact equality. validate_effective_phase0_bundle and downstream gate validators consume this chain.

Changing pressure['anchor_y'] from a shared dynamic scalar to a device [world] tensor without versioning this chain would make old and new snapshots ambiguous, omit the rowwise reset/runtime target contract from the authoritative record, and allow a report/config consumer to interpret a dynamic value as a static constructor setting. The schema owner, migration policy and tests must therefore be approved before implementation.

## Objective and inherited requirements

This document supersedes the per-world pressure-anchor plan and inherits its full scope and semantics. It also inherits all previous Phase 4 contact/start-profile/focal-pose/reset requirements and the entire all-five rollout/candidate/route/terminal-status/fixed PPO plan: A1/A2/A3/C1/C2 distributions, D1/D2 matched controls, candidate identity/order/recall, execution failure actor eligibility, replay digest/schema, GAE, exact update cadence, optimizer/RNG/checkpoint continuity, and two-WX9100 three-pass/runtime gates. No earlier requirement is waived.

Extend provenance so a static effective snapshot declares the dynamic pressure-anchor contract without embedding world-specific runtime measurements. A completed episode report stores actual per-world anchors under a distinct runtime report schema and binds them to reset generation/source identity. Static pressure parameters and protocol assignment remain static. Version and validate snapshots, receipts and baseline freezes deliberately, while giving historical v1 artifacts a safe, explicit compatibility path.

## Schema owner and consumer audit

1. frc_defense/meco_baseline_freeze.py owns EFFECTIVE_SNAPSHOT_SCHEMA, EFFECTIVE_FREEZE_SCHEMA, exact snapshot shape, receipt/freeze validation, source path closure, Phase 0 bundle validation and downstream callers. It must define current v2 snapshot and v4 freeze contracts and an explicit archive-only v1 validation route; no code may silently parse a v1 artifact as v2.
2. AUTHORITATIVE_FIELD_SOURCES closes runtime value-owner hashes for eight effective fields. The schema source itself must also be hash-bound in the capture receipt. Add an explicit snapshot-schema source manifest/path set for meco_baseline_freeze.py, and update receipt compilation/validation to require exactly that closure, without falsely claiming the file computes the pressure numeric value.
3. frc_defense/curriculum_scenarios.py::_effective_curriculum_snapshot emits the v2 dynamic runtime-state contract alongside static resolved_fields and canonical config artifacts. The v2 contract declares pressure anchor schema ID, [world] shape, dtype/device ownership, capture stages, selected-row reset semantics and that actual numeric anchors are not static config. Static pressure config/artifact must no longer include a runtime anchor scalar.
4. compile_effective_capture_receipt and validate_effective_capture_receipt must reproduce v2 snapshots exactly from factory arguments, source identity, new schema-owner hash, protocol bytes and runtime identity. A resealed but nonreproducing anchor contract fails closed.
5. compile_effective_baseline_freeze, validate_effective_baseline_freeze, validate_effective_phase0_bundle, paired calibration gate consumers and any artifact-schema constant consumers must explicitly bind the v4 freeze hash. No old freeze hash is reinterpreted as v4.
6. meco_track_training.py's completed-episode report adds per-world pressure anchor values, anchor schema, reset-generation identity and source binding as passive evidence. It must never serialize only world zero or feed the report back into current policy, reset, qualification or campaign control.
7. Existing v1 captures/receipts/freezes may exist externally and are immutable. New compilation emits v2 snapshot/v2 receipt/v4 freeze. V1 remains valid only through an explicit archive-integrity validator that checks its original canonical schema/digests and is marked archived/non-authoritative; it cannot produce a new v4 freeze or pass current Phase 0/current-source qualification. Migration requires a fresh v2 capture from the original exact factory arguments and currently attested source/runtime; never edit or reseal a v1 artifact as v2. If exact archive integrity cannot be established without old source content, return a clear archived-unverifiable status rather than treating it as current.
8. tests/test_meco_baseline_freeze.py exercises snapshot exact keys, source closure, receipt reproduction, freeze/Phase 0 binding, config byte integrity and source drift. Add v1 archive-only acceptance/rejection and v2 current-flow tests without weakening current-source checks.

The v1 artifact rule is deliberate: preserve forensic validation of existing canonical bytes and recorded hashes, but prevent old row-zero semantics from authorizing a current device-resident run. No silent migration or qualification path exists.

## Complete authorized schema/report scope

Every target and requirement in the superseded pressure/reset/PPO plan remains authorized and mandatory. This expansion adds:

| Target | Authorized work | Boundary |
|---|---|---|
| frc_defense/meco_baseline_freeze.py | Define separate v1/v2 effective snapshot, v1/v2 capture receipt and v3/v4 freeze schema constants; emit/validate current v2/v4; implement explicit v1 archival integrity validation; hash-bind schema-owner files; update source closure, exact top-level keys, canonical digest and Phase 0/freeze compatibility checks. | Historical v1 is archive-only/non-authoritative; never auto-upgrade, accept as v2, or use to compile current freeze/campaign. Keep final-test/allocation authorization semantics unchanged. |
| frc_defense/curriculum_scenarios.py::_effective_curriculum_snapshot | Emit dynamic runtime-state contract and static pressure config with no row-derived anchor scalar. Include schema contract in canonical snapshot/field identity and config artifacts as appropriate. | Snapshot-only construction remains environment-free; do not read GPU/world runtime tensors to create static snapshot. |
| frc_defense/meco_track_training.py | Emit actual per-world anchor vector, stage identity, reset generation, schema and source identity in the passive completed-episode record. Update record schema as required. | Serialization only after decisions; never collapse heterogeneous values to row zero or re-enter device/control path. |
| tests/test_meco_baseline_freeze.py | Cover exact v2 snapshot schema and dynamic-state contract, source schema-owner path closure, reproduction, v4 freeze binding, Phase 0 rejection of legacy artifacts, v1 archive-only validation, v1-to-v2 tampering, and missing/extra schema fields. | Do not remove existing source, runtime, canonical bytes, config digest, or snapshot reproduction failure tests. |
| tests/test_meco_phase4_pressure_anchor.py (from predecessor) | Assert static snapshot has only the anchor schema contract; episode report has actual per-world anchors and never a scalar row-zero projection. | Synthetic/schema assertions only. |
| tests/test_meco_phase4_runtime_adapter_contract.py and tests/test_meco_pressure_anchor_schema.py (new) | Verify report schema/version, reset/source binding, static-vs-dynamic distinction and safe historical status. | No policy/qualification control from host report fields. |

No edits are authorized to unrelated snapshot fields, tuple partition algorithms, trusted allocation/test approval constants, scoring estimands, configuration artifact semantics other than removing dynamic pressure anchor and adding its static schema declaration, or campaign eligibility policy. If source tracing identifies another consumer that must change, stop for a new exact-file Pass 1.

## V2 snapshot and runtime report contract

1. Current snapshot schema is v2 and has a strict, documented exact top-level key set. It declares a dynamic runtime-state manifest with a pressure-anchor record containing schema ID, shape [world], floating dtype, device ownership, source/capture stage names, reset mask semantics, and static/dynamic separation. No actual pressure-anchor values appear in the static snapshot.
2. The static pressure value under resolved_fields.opponent_controllers and the opponent_schedule config artifact contains only category/family/slots/phase/style and other immutable configuration. Remove or rename the old anchor_y field; no consumer may use it as runtime target input. V2 includes a contract marker that these values derive per row from the selected world pose at the two predecessor reset stages.
3. Current capture receipt uses a strict v2 receipt schema embedding a v2 snapshot; the snapshot producer source identity and authoritative hashes include the schema owner. V1 receipt/snapshot validation is a distinct archival-only path and returns an explicit status preventing use by current compile/Phase 0 APIs.
4. Current baseline freeze uses v4, binds the v2 receipt digest and schema-owner source, and is the only freeze emitted for fresh captures. V3 can only be checked as archival legacy evidence paired with its v1 receipt; it cannot be re-signed as v4.
5. All downstream consumers compare exact snapshot/freeze schemas and hashes. No consumer may accept arbitrary schema strings, ignore the dynamic-state manifest, or normalize v1 fields into a v2 report.
6. Completed episode report uses a versioned pressure-anchor record with the complete per-world vector, selected world identifiers/mask coverage, capture stage, reset generation, schema and source identity. The vector is materialized only at the passive report boundary. A homogeneous vector is still serialized as a vector; no conditional scalar shortcut is needed.
7. If current scenario output requires a legacy scalar for an unrelated reader, the reader must remain explicitly pinned to archival v1 records; no row-zero scalar is emitted for new v2 records.
8. Snapshot/report/resolved-field/config-artifact schema hashes participate in source identity, PPO replay/checkpoint identity and episode provenance. Incompatible schemas fail closed before any parameter, optimizer or campaign state mutation.

## Migration and failure behavior

- New captures: v2 snapshot + v2 receipt + v4 freeze; compile and validate only against the exact current source tree and attested runtime.
- Existing v1 artifacts: validate their original schema and integrity through the v1 archival API; return legacy/archived, qualification-ineligible status. They remain byte-for-byte unchanged and cannot enter current Phase 0 or make a v4 freeze.
- Re-capture: to migrate a configuration, provide the same explicit factory arguments under the current source/runtime and compile a new v2 capture. The new snapshot hash necessarily differs because schema and pressure-state contract are different. Preserve a link to the old archived digest in external provenance only; do not make it part of numeric pressure config.
- Tampering, missing schema-owner hash, mismatched v1/v2 fields, forged pressure anchor values in static snapshot, unsupported report version, ambiguous scalar conversion or changed current source must fail closed. Do not fall back to row zero or permissive legacy parsing.

## Acceptance, tests and stop conditions

Accept only if v1 archive records remain explicitly integrity-checkable but cannot authorize current execution; v2 current snapshot/receipt/freeze reproduce and hash-bind the exact source/schema owner; static snapshots contain a contract but no runtime anchor scalar/vector; reports carry per-world vectors and correct generation/source; all Phase 0/downstream validators reject wrong versions; and all previous homogeneous/mixed anchor, reset isolation and five-track PPO gates remain intact.

Stop before editing if an existing v1 consumer cannot be classified archival versus current; if v1 validation needs silently accepting changed source; if v2 compiler still emits an anchor scalar in static config; if v2 validation does not bind meco_baseline_freeze.py; if freeze/report/Phase 0 consumers require unspecified files; if a v1 artifact can enter current qualification; if a new source target is needed; or if per-world numeric anchors reach static hashes/config through host extraction. Retain all predecessor stop conditions for row isolation, RNG/checkpoint, Phase 4/focal behavior, host synchronization, candidates, policies, replay, optimizer cadence and campaigns.

Schema and report tests above augment, but do not weaken, every prior pressure-anchor/reset/profile/focal/PPO test. Tests asserting simulated behavior still require the AGENTS-mandated representative seeded end-to-end simulator evaluation and fresh inspected replay after Pass 3 and WX9100 release; verify current dashboard backend and frontend before replay.

## Sequential three-pass gates

1. Pass 1: independent review approves exact artifact/head/hash; schema-owner source closure; v1 archival/v2 current migration; v2 contract contents; v4 freeze and downstream consumer compatibility; report schema; all predecessor requirements. No implementation/runtime work.
2. Pass 2: only after approval, use fresh HDD worktree from latest main with the approved plan lineage. Publish final file/schema/consumer inventory before edits; commit/push incrementally. No tests/imports/compilation/GPU/simulator/dashboard/replay/training/campaign while either WX9100 lease is held. New target returns to Pass 1.
3. Pass 3: independent exact-head read-only review of source, validators, v1/v2 compatibility, current-source closure, anchor/report schema, reset per-row identity and all five policy/candidate/PPO invariants.
4. Only after Pass 3 approval, explicit user stop and verified release of both WX9100 leases may approved tests run on attested WX9100s. Any simulated-behavior assertion also requires the current-checkout dashboard/fresh replay validation required by AGENTS.md.
5. This plan does not start, resume or alter any training/evaluation campaign.

No implementation, tests, imports, runtime, GPU, simulator, dashboard, replay, training, fitting, evaluation or campaign operation was performed to create this plan. This document does not assert that snapshot v2, per-world pressure anchors, selected reset, or all-five GPU residency currently exist.
