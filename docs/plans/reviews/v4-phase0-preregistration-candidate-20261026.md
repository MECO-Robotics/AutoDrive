# Independent final Pass 3 review: P0a Phase 0 preregistration candidate

- **Verdict:** ACCEPT — exact candidate plus authorized maintainer decision record
- **Review date:** 2026-10-26
- **Reviewer identity:** `plan_c_owner`
- **Reviewer task:** independent final exact-head Pass 3 review assigned by `/root`
- **Candidate author:** `p0a_implementer` (not reviewer or decision authority)
- **Maintainer decision authority:** user / authorized maintainer; decision record author `Brian Lee`
- **Reviewed candidate commit:** `56183cefd2402342ce28f836155164fc850ebc60`
- **Reviewed candidate file:** `docs/plans/v4-phase0-preregistration-candidate-20261026.md`
- **Candidate blob:** `ad579cee66e69ddf570041a483c4daa3176c402c`
- **Candidate file SHA-256:** `c8ace83e0e5a3fdef8d33ab090d3cbb9193e12342b015d72b07016acdb603726`
- **Maintainer decision record commit:** `7007607010636fe7451ad1b59eb1a900e0a4d7b1`
- **Decision file:** `docs/decisions/meco-v4-p0a-maintainer-decisions-20261026.md`
- **Decision blob:** `0d71236ed1ff0c8b0c5ca875eafbf099c04a8611`
- **Decision file SHA-256:** `d9394b07bc508021a4b10faf89b2d258a863183288d633442f796fc181592e64`
- **Recorded approval:** authorized maintainer reply “Approve all,” dated 2026-10-26. The record explicitly says its exact reply timestamp is unavailable and asserts no cryptographic signature.
- **Accepted P0 package plan:** commit `3071c2496eb8bcb326ad0f1e971f1bd1e8d0858f`, file SHA-256 `5ac9e99cf0ead523cecf6cdc64753d7801361fc876b862f762ae18f70ef4c831`; exact-head Pass 1 ACCEPT record `f2f299a81d266dc5e08c5ce27402151c0e78571d`, record SHA-256 `baf8666f6e93550f31afc441e87eaeb089dffd6e4401376bb943ca978e4d529e`.
- **Accepted P6 dual-protocol plan:** commit `c1f20c80a15588725b3f346ae06366c2197465b7`, file SHA-256 `8c122a5ea82e2f3b77f222c8954a92ae6d11e54a4ded55bfdb786d42283fc516`; exact-head Pass 1 ACCEPT record `f025827abe2d3aec366b65b4fad85debd5de31aa`, record SHA-256 `821a11b053c86780bca35aff45e53373c003d7046468e4e1b6905d39d67df6c8`.
- **Accepted overall plan:** commit `8efec9378d1912f0839d7c6cd8e88467fceb1d5a`, SHA-256 `57f20a1bcfe6f229429d402140de93a0ee1780372e21a9ac3344f81c72faca90`; ACCEPT record `b0b2962d299b1ecaab053b873ef2ae09e3c5f467`, record SHA-256 `54e68870b3b95fd91b92d3f934e44d5ec9bb63e1fb6d03f67e6d95f10d3abb4a`.
- **Source reference base:** `6ed961a09c08dfc1796d98ae3566bd3f0eaaaf22`.
- **Source evidence pins:** score ledger `frc_defense/meco_simulated_score_ledger.py` at `d74b0a15e25ddbc6bacfbbb864674371629bb19e`, SHA-256 `71427d251f1639a4a113b610b5c8122fe88d1c2aac9c11eb4861b0f82bc05114`; safety metric ledger at the same commit, SHA-256 `1d370591e5aa76ffc8b891a691268e3d897169c4dfd5456b282141c8596ad2f1`; Stage 1 qualification source at the same commit; curriculum source base `6ed961a...`.

## Review findings

The decision record names the exact candidate commit and file SHA under review and states that “Approve all” applies to that candidate's source-defined score proxy, both finite case matrices, and proposed analysis, safety and failure rules. I checked the record against every §10 row:

- **Proxy and interpretation:** approves whole-alliance `simulated_match_score`, sum of score units over unique authoritative accepted active-HUB FUEL award event IDs, red/blue totals, no per-robot causal attribution, and the limited simulated interpretation/exclusions. This matches the pinned score ledger's formula, duplicate-ID checks, alliance mapping, red/blue aggregation, and null official-points field (`meco_simulated_score_ledger.py:95-104,287-314,359-378`), plus the cited ledger design (`phase0-score-ledger-replay-implementation-20261025.md:42-46`).
- **Dual target roles:** approves `pickup_simple` as the sole Stage 1 protocol and `full_match_offense` as the separate primary score protocol, with cross-binding and no substitution. The decision preserves the P6 source/API implementation and review gate before P0b.
- **Finite matrices and identities:** approves both the two-category Stage 1 matrix and the full-match matrix, the explicit v5 layout override, pressure-only defender mapping, held-out `split_chaser`, and paired alliance assignment. It also approves partition disjointness, 100 episodes per category/stratum for calibration/validation/final test, at least two calibration repetitions, deterministic seed construction and paired seeds. Exact expanded tuples/seed bytes and opponent policy/checkpoint/config identities are explicitly deferred to be source-bound before any new candidate training or outcome collection. These are deterministic protocol materializations and implementation identities, not unresolved choices of the finite matrix; the candidate requires freezing them before outcomes.
- **Statistical choices:** approves 10,000 paired-block bootstrap replicates, within-stratum resampling, keeping policy rows together, Bonferroni familywise 95% coverage over five track contrasts, and two-sided per-contrast 99% percentile intervals at the 0.5th/99.5th percentiles. Calibration-derived numeric gains/margins remain deferred to P0b as the accepted P0 plan requires; they are not chosen after final-test outcomes.
- **Safety:** approves the three source-defined focal-slot contact-rate metrics and calibration-derived noninferiority margins, separately from the four proposed hard veto classes. It requires source-bound signals before claiming a safety pass. The candidate correctly states illegal-transition and footprint signals are currently unavailable (`meco_metric_ledger.py:220-235,287-302`); source/API implementation and validation remain a required pre-claim gate, not an unresolved approval choice.
- **Failure and comparator rules:** approves the per-class handling, no replacement, no zero-imputation for incomplete score episodes, Stage 1 timeout failure, and D0/D1/D2 with D3 excluded absent a prior amendment.

The decision record is properly bounded: it records a user approval and no cryptographic signature, approves protocol choices only, and grants no tests, runtime, calibration, GPU, dashboard, training, integration, merge, promotion, or campaign operation. The current score-ledger approval's Phase 0 digest still must be reconciled to the accepted P0a/P0b contract before outcome collection as the candidate requires. Source implementation, exact case/seed bytes, opponent identities, numeric calibration margins, P6 schema/API conformance and operational G0 evidence remain later gates with explicit ordering before collection/training/qualification; they do not leave a §10 product choice unresolved.

The exact P0/P6/overall plan pins and their ACCEPT records were verified. Stage 1 `pickup_simple` target and closure/cases match current source; the score-evaluation target and source-grounded case matrices remain separate. The 160-physics-tick horizon, macro cap/partial final macro, gamma and continuation exponent convention remain consistent with the accepted profile.

No material §10 choice remains unresolved for P0a. Exact seed bytes and identities are frozen as protocol implementation outputs before outcomes, while numeric margins and source/API/runtime validation are correctly deferred to P0b/post-G0 gates. Therefore the exact candidate plus this exact decision record satisfies P0a Pass 3. This ACCEPT applies only to these two exact documents and their pinned sources.

## Authority and review limits

The human decision is attributed only as recorded in the decision document; this reviewer does not claim to be the approver or to provide a cryptographic signature. ACCEPT covers preregistration documentation only. It does not certify source implementation, safety signal availability, calibration results, P0b artifacts, runtime/GPU/dashboard freshness, training progress, or Stage 1 qualification.

Read-only document/source review. No tests, builds, runtime, simulator, calibration, GPU/device, dashboard, campaign, training, checkpoint, or output operations were performed.
