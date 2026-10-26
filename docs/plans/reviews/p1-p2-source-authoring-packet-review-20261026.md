# Independent review: P1/P2 source-authoring readiness packet

- **Verdict:** REVISE
- **Reviewed packet commit:** `1562106255ba174de3e0a8e7a917cfebee9c6e33`
- **Reviewed file:** `docs/plans/p1-p2-source-authoring-packet-20261026.md`
- **Packet SHA-256:** `db197f0be3284391eb01dd6c0ecc0b36373bec524d8ba4db6d75bfeed9e5f13e`
- **Review branch base:** `d74b0a15e25ddbc6bacfbbb864674371629bb19e`
- **Review scope:** factual support, current package-plan pins, and whether the readiness packet accurately states source-only gates. No source or execution work was performed.

## Findings

### F1 — P1 package plan exists and is accepted, but its source gate conflicts with the newer overall plan

Packet lines 8–10 and 18 say the scoped P1 plan/allowlist is absent and propose that a new Pass 1-accepted plan plus role assignment is the first P1 source-start gate. An exact P1 package plan already exists: `53686a254318484f9c6bc3496ecc1c461572c53a`, `docs/plans/meco-five-track-v4-schema-score-artifact-implementation-plan-20261026.md`, SHA-256 `21c07b2c08d37e48ed5d0dac1f149e4a45eb89d59360f466b587f534cf81e076`; its independent Pass 1 ACCEPT is recorded at `854d5286947074ed33f7ac059d67f2f9ccbf9257` (record SHA-256 `0680a57e5ff0427b20a0a5bd86c224085583b241b3f57bcb18bde69cab0f158f`). It includes a finite file allowlist at lines 313–338.

That accepted P1 plan still pins the superseded overall plan `632592…` and explicitly makes G0 a prerequisite to P1.1 source discovery and implementation (plan lines 72–82, 179–183, 410–412). The current accepted overall plan is `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; it allows isolated pre-G0 source/test-file authoring after the applicable package plan and prerequisites are accepted. The package's stricter old gate has not been amended. Therefore the packet's proposed start condition is incomplete: P1 has an accepted package plan, but it does not yet authorize pre-G0 source work under the current overall plan. Revise the packet to cite the existing exact plan/allowlist and identify a P1 plan amendment with a fresh exact-head Pass 1 review as the gate for any pre-G0 source-only slice. Keep the P1 candidate API/SOURCE/API ACCEPT as a separate downstream gate for P2.

### F2 — P2 plan disposition is stale

Packet lines 19 and 28 cite the predecessor P2 branch head `fa15258776030ccb2ca5f9f637a38594075797d` and characterize its corrected gate as pending review. A corrected successor had already been authored and accepted before the packet's commit time: plan `09cf081a4b47b66a9b3fcecf3c3748eda5722465`, file `docs/plans/meco-v4-p2-gpu-resident-environment-route-package-plan-20261026.md`, SHA-256 `eb034463a5c4d0c655fe8e16d7636247ab899ec56bca1c4655122f725c5041d1`; exact Pass 1 ACCEPT record `de260bde20b30368d84531767cc0e8d3bf072b35`. The plan commit is timestamped 05:06 UTC and its ACCEPT record 05:07 UTC; the readiness packet is timestamped 13:39 UTC. Update P2's plan/status pins accordingly. P2 remains blocked on the exact P1 candidate API/source head receiving independent pre-G0 SOURCE/API ACCEPT, plus distinct assigned roles and its own stated prerequisites; accepted P2 plan status alone does not satisfy those gates.

## Supported scope and limits

The packet correctly treats P0a approval as protocol approval rather than API acceptance, keeps runtime/integration/GPU/dashboard/campaign actions gated, and says exact case/seed bytes and opponent identities must be frozen before new candidate training or outcome collection. Its overall “no P1/P2 source slice yet” conclusion can remain, with the two gate corrections above. No evidence in this review supports P1/P2 implementation API acceptance, source conformance, G0, or runtime readiness.

No tests, builds, imports, simulator, dashboard, GPU/device, campaign, or lease operations were performed. This review applies only to the exact packet commit and hash above.
