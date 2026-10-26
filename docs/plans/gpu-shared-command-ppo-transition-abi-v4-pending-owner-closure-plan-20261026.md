# PPO Transition ABI V4 Pending Owner Closure and Reconciliation Plan

Date: 2026-10-26  
Status: **Pass 1 planning candidate; six owner rows remain PENDING; no ABI adoption or implementation authorization**

## Purpose and scope

This plan defines the next documentation-only tranche for closing the six
pending owner rows in the proposed shared command/PPO transition ABI V4 and,
afterward, reconciling compatible owner decisions into one immutable profile.
It does not make technical decisions for those owners, infer human maintainer
approval, adopt V4 as an ABI, authorize source/schema/runtime work, or change
the active training campaign.

The authoritative proposal and matrix inputs are pinned here:

| Artifact | Commit | SHA-256 |
|---|---|---|
| V4 proposal, `docs/plans/gpu-shared-command-ppo-transition-abi-v4-20261026.md` | `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b` | `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6` |
| V4 owner-adoption matrix, `docs/plans/gpu-shared-command-ppo-transition-abi-v4-owner-adoption-matrix-20261026.md` | `ee234b2c5162038e6fcfff6d9551d80f443f6d82` | `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53` |

V3 remains immutable. The current V4 artifacts contain recommendations and
owner decisions, not a published normative contract. Neither the accepted
delegated-response workflow nor an owner-row **ADOPT** is human maintainer
approval or implementation authorization.

## Current independently reviewed owner-response evidence

The following response heads and blob hashes were independently checked from
their pushed origin branches for this plan. The Pass 3 verdict and reviewer
identity are recorded in the orchestration thread. There are no persisted
review-record files or review artifact hashes yet. The next tranche must
persist one reviewer-authored record for every row below, including the three
rows whose owner disposition is already ADOPT, before reconciliation proceeds.

| Role | Response path | Pushed response commit | Response SHA-256 | Exact-head Pass 3 record | Current disposition |
|---|---|---|---|---|---|
| Environment/profile | `docs/plans/v4-owner-responses/environment-profile.md` | `cd9276c62edb695715bf54ef44dcd7703183ee5b` | `e0cac32ea061f73b96b57d378c35784f73643ea3b2dfb6fa3ca1a65e56d47a2a` | ACCEPT, reviewer `plan_a_owner` | ADOPT exact pinned V4 row |
| Plan A | `docs/plans/v4-owner-responses/plan-a.md` | `03b7a9f5cbb161a2b30563e76a360619df144849` | `e6863a2b3b52fc2526062ac081cec0b153742d0f32344d8006d766cfad74df21` | ACCEPT, reviewer `plan_c_owner` | ADOPT exact pinned V4 row |
| Plan C | `docs/plans/v4-owner-responses/plan-c.md` | `5b49efb8405f3359306715427bd71996b37429d4` | `8ca11c3987ed34cf7b13ef41b0abecb6e33a703a9c3de92c3eed0f962d8e0232` | ACCEPT, reviewer `plan_a_owner` | PENDING |
| Route | `docs/plans/v4-owner-responses/route.md` | `cc2798dbe2b6f842f764231b3d457157a9377fd3` | `3ccd83285d223d8cade02224962db24646a6d3b8de7a50560e8ba05495ed0ce0` | ACCEPT, reviewer `plan_c_owner` | PENDING |
| Observation/native | `docs/plans/v4-owner-responses/observation-native.md` | `94bd25a2456de90d4c4119473adc8f04f02256b4` | `f114fcac3f98a685c0e616b426ee01ef0741d01c0bbf8ee918e028f9ca49328f` | Corrected head ACCEPT, reviewer `plan_c_owner`; initial head was REVISE | PENDING |
| PPO/RNG | `docs/plans/v4-owner-responses/ppo-rng.md` | `4254f45eca5f2e938000b6e0dda4348ad4ea9337` | `3f1a05cf9bc4434f413d13d666aa899ad52c4922e46f0390604c87098fa04c10` | Corrected head ACCEPT, reviewer `plan_a_owner`; initial head was REVISE | PENDING |
| Gamepiece/score | `docs/plans/v4-owner-responses/gamepiece-score.md` | `3738ad58fc883c5853b32c508e89ac657864bca4` | `f57476d13d6038e9e0109fedc557c617ead25a99e4db49512b135a532c1e13af` | Corrected head ACCEPT, reviewer `plan_c_owner`; two prior heads were REVISE | ADOPT exact pinned V4 row |
| Artifact/manifest | `docs/plans/v4-owner-responses/artifact-manifest.md` | `da5639605f262a4e04b6cec99afc0f7391160512` | `d2df36aedc08d3ff9c5fc34c75405faab34bbc9af8a5e896496109d21f434fe7` | ACCEPT, reviewer `plan_c_owner` | PENDING |
| Evaluation/curriculum | `docs/plans/v4-owner-responses/evaluation-curriculum.md` | `1ee6932d16245ce99c16009e9d96adb037c05626` | `346bc926a485b48aadb0466d51a9c40d3b8fa5945bbce4c78fc6bcfe0975a4ab` | ACCEPT, reviewer `plan_a_owner` | PENDING |

The three ADOPT dispositions are limited to their respective rows. Six rows
remain PENDING despite their response documents passing editorial/contract
review. A review ACCEPT does not convert a PENDING owner disposition into
adoption. The thread-recorded reviews do not satisfy the persisted-record gate
below, and this plan does not claim that any review record already exists.

## Pass 2: close six pending rows

After independent Pass 1 acceptance of this plan, assign one delegated
technical owner to each pending row. Each owner may edit only that row's
existing response path in a separate branch/worktree, and must commit and push
the final response. The response must choose either **ADOPT** of the exact V4
proposal and matrix hashes above, **REJECT-WITH-REPLACEMENT** with an immutable
versioned replacement artifact and full SHA-256, or remain **PENDING** with
specific unresolved evidence/decision gates. No V4 proposal or matrix edits are
allowed in this tranche.

| Owner row | Sole Pass 2 response path | Minimum closure evidence (do not fill values by inference) |
|---|---|---|
| Plan C | `docs/plans/v4-owner-responses/plan-c.md` | Resolve the response's stated candidate/capacity and continuation semantics against authoritative C-route source and ABI consumers; cite exact source/contract evidence and specify all cross-row effects. If source does not establish a product choice, keep that choice explicit as a decision gate. |
| Route | `docs/plans/v4-owner-responses/route.md` | Resolve route field and geometry/validity semantics against authoritative route source and consumers; identify dependencies on Plan C, environment/profile, and observation/native. Do not infer route shape or legal-value policy from a recommendation. |
| Observation/native | `docs/plans/v4-owner-responses/observation-native.md` | Resolve row identity, generation/lifetime, mapping and stale-handle behavior using authoritative native producer/consumer contracts. Explicitly gate unresolved slot reuse, generation width/wrap, and invalidation semantics rather than inventing values. |
| PPO/RNG | `docs/plans/v4-owner-responses/ppo-rng.md` | Resolve update atomicity, valid-action/bootstrap semantics, RNG stream ownership/state and failure/replay boundaries against authoritative trainer/PPO code and required reproducibility contract. Keep choices not established by current contract as owner/product decision gates. |
| Artifact/manifest | `docs/plans/v4-owner-responses/artifact-manifest.md` | Define, with evidence or explicit owner decision gates, immutable identity, compatibility/version binding, provenance, required fields, and failure behavior needed to bind the adopted profile to checkpoints and downstream consumers. Do not claim an artifact schema exists where it does not. |
| Evaluation/curriculum | `docs/plans/v4-owner-responses/evaluation-curriculum.md` | Bind qualification/evaluation identity, protocol/version, episode/layout/category coverage, score estimand, pass predicates and checkpoint/artifact references to authoritative curriculum requirements. Any missing gate or criterion remains a named decision gate; diagnostic results cannot be promoted to qualification evidence. |

Each owner response must state role/author/date, immutable input hashes,
disposition, evidence with exact citations, unresolved decisions/dependencies,
and that it records only a delegated technical decision—not human maintainer
approval, another row's consent, ABI publication, or implementation
authorization. A **REJECT-WITH-REPLACEMENT** is complete only if its replacement
is immutable, hashed, versioned, and includes affected field definitions and
payload impact; unsupported rejection stays PENDING.

### Replacement impact and re-decision rule

If any owner proposes a replacement, no other row is presumed to cover it.
The replacement must list its field-level changes and dependency set. Every
owner row whose semantics, byte layout, offsets, validity rules, lifecycle,
producer/consumer contract, or qualification binding depends on a changed
field must issue a new response pinned to the exact replacement hash and
receive a new independent exact-head Pass 3 review. The owner matrix remains
PENDING for affected rows until those re-decisions are reviewed. If impacts
cannot be bounded from evidence, treat all potentially dependent rows as
affected; do not silently carry forward V4 decisions.

### Per-response review gate

The only permitted review-record paths are these nine exact files:

| Row | Sole review-record path |
|---|---|
| Environment/profile | `docs/plans/v4-owner-responses/reviews/environment-profile.md` |
| Plan A | `docs/plans/v4-owner-responses/reviews/plan-a.md` |
| Plan C | `docs/plans/v4-owner-responses/reviews/plan-c.md` |
| Route | `docs/plans/v4-owner-responses/reviews/route.md` |
| Observation/native | `docs/plans/v4-owner-responses/reviews/observation-native.md` |
| PPO/RNG | `docs/plans/v4-owner-responses/reviews/ppo-rng.md` |
| Gamepiece/score | `docs/plans/v4-owner-responses/reviews/gamepiece-score.md` |
| Artifact/manifest | `docs/plans/v4-owner-responses/reviews/artifact-manifest.md` |
| Evaluation/curriculum | `docs/plans/v4-owner-responses/reviews/evaluation-curriculum.md` |

Each record must be authored by an agent distinct from that response's author
and must review the exact response commit and blob SHA-256 pinned in the table
above. This applies to all nine current heads, including environment/profile,
Plan A, and gamepiece/score, which already have reviewed ADOPT dispositions.
Each record must give reviewer identity, exact response commit and blob SHA,
verdict, review scope, evidence checked, and whether the owner disposition is
supported. A thread-recorded verdict is useful provenance but is not the
persisted review record required for closure. An ACCEPT may validate a PENDING
response while leaving its owner row PENDING. REVISE requires a corrected
response head and a new record at that row's same review path, superseding the
older record; review paths cannot be duplicated or expanded. No row advances
on an unreviewed or superseded response head.

## Shared reconciliation and payload gate

After all nine current owner rows have final independently reviewed
dispositions against compatible immutable inputs—and after every replacement
impact has been re-decided—create a separate versioned reconciliation artifact
at:

`docs/plans/gpu-shared-command-ppo-transition-abi-v4-owner-reconciliation-20261026.md`

The reconciliation must list all nine exact response commits and response
hashes, each disposition, all nine persisted Pass 3 review-record paths and
their exact blob SHA-256 values, all replacement artifact hashes, and a
field-by-field compatibility table. It must distinguish adopted
values from explicit unresolved/product decision gates. A shared ABI/profile
cannot be published as adopted while any required field or owner row remains
PENDING or mutually incompatible. No human approval is implied; any required
maintainer approval must be obtained through the repository's actual governance
process and recorded separately.

Recompute the complete payload and envelope from the reconciled field types,
dimensions, bounds, alignment/padding, capacities, and versioned metadata.
Record the derivation for every field and total, including per-row bytes,
batch bytes, auxiliary payloads, framing/envelope bytes, and any conditional
profile assumptions. Cross-check arithmetic independently. Do not copy V4's
proposal totals if an adopted replacement or owner choice changes any input.
Pin the resulting reconciliation artifact SHA-256 and final profile/ABI hash;
the two hashes must identify unambiguous immutable bytes. Any subsequent field
change requires a new reconciliation version and a full recomputation.

An independent Pass 1 plan review authorizes the reconciliation artifact to be
prepared. After it is prepared, a distinct independent exact-head Pass 3
reviewer must verify owner/replacement coverage, compatibility, decision-gate
closure, byte derivation, immutable hash binding, and absence of unsupported
human-approval or implementation claims. Persist that review. Any revision
changes the candidate head and requires review of the new exact head. The
owner-adoption matrix may be updated only in a separately scoped, reviewed
documentation tranche that records the reconciled evidence; no matrix mutation
is authorized here.

## Explicit boundary and next implementation gates

This plan authorizes documentation-only Pass 2 owner responses after its own
Pass 1 ACCEPT. It does not authorize changes to V4, the matrix, integration
ledger, schemas, source, tests, compilation, runtime/device execution,
simulator/dashboard/replays, checkpoints, campaign state, training, evaluation,
or qualification.

Even after a reconciled immutable ABI/profile is independently reviewed, no
source or schema work may begin until a separate implementation plan is
written against that exact adopted ABI/profile hash and passes independent
Pass 1 review. Schema implementation and source implementation then require
their own scoped Pass 2 changes and exact-head independent Pass 3 reviews;
any test, simulator, dashboard, device, evaluation, or qualification work
must follow its own reviewed scope and applicable project validation rules.

The continuous training boundary remains in force: do not stop, restart,
reconfigure, evaluate through, or otherwise touch the active campaign, GPUs,
GPU leases, workers, checkpoints, or dashboard. This documentation plan makes
no claim that any track has passed Stage 1. Diagnostic campaign evidence is
not qualification evidence, and this tranche cannot establish the goal's
five-track Stage 1 completion condition.

## Pass 1 review scope

The sole file allowlist for this planning candidate is this document. Its
review checks factual pins, scoped Pass 2 paths, replacement/re-decision logic,
payload/reconciliation gates, no-human-approval language, and the explicit
source/runtime/training boundary. Pass 1 ACCEPT authorizes documentation-only
work on exactly the six response paths in the Pass 2 table and exactly the nine
review-record paths listed above. The six response paths and nine review
paths are separate allowlists; a response author may not write its review
record. No other file or action is authorized by this plan.
