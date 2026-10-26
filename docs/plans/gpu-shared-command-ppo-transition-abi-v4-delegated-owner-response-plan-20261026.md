# PPO Transition ABI V4 Delegated Owner Response Plan

Date: 2026-10-26
Status: **Pass 1 planning candidate; owner responses not started; all dispositions PENDING**

## Purpose and pinned evidence

This document defines a documentation-only process for collecting nine scoped
owner decisions about the proposed shared command/PPO transition ABI V4. It
does not adopt the proposal, define an implementation ABI, or authorize
schema/source changes.

The immutable inputs for every response are:

| Artifact | Commit | SHA-256 |
|---|---|---|
| V4 proposal, `docs/plans/gpu-shared-command-ppo-transition-abi-v4-20261026.md` | `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b` | `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6` |
| V4 owner-adoption matrix, `docs/plans/gpu-shared-command-ppo-transition-abi-v4-owner-adoption-matrix-20261026.md` | `ee234b2c5162038e6fcfff6d9551d80f443f6d82` | `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53` |

The matrix names the affected owner domains and records the proposed values as
recommendations. Its nine owner dispositions and shared ABI publication gate
remain **PENDING**. This plan and its later responses preserve that status
until a response has received its own independent exact-head review.

The accepted V4 proposal is a versioned successor and leaves V3 immutable.
Neither V4 proposal acceptance nor acceptance of this workflow establishes
owner adoption. Any V3 contract, source, or existing plan remains unchanged.

## Nine scoped response assignments

Each role may decide only its corresponding V4 matrix row. The only permitted
response paths are the following exact files:

| Owner role | Sole response path | Initial disposition |
|---|---|---|
| `environment_profile_owner` | `docs/plans/v4-owner-responses/environment-profile.md` | PENDING |
| `plan_a_owner` | `docs/plans/v4-owner-responses/plan-a.md` | PENDING |
| `plan_c_owner` | `docs/plans/v4-owner-responses/plan-c.md` | PENDING |
| `route_owner` | `docs/plans/v4-owner-responses/route.md` | PENDING |
| `observation_native_owner` | `docs/plans/v4-owner-responses/observation-native.md` | PENDING |
| `ppo_rng_owner` | `docs/plans/v4-owner-responses/ppo-rng.md` | PENDING |
| `gamepiece_score_owner` | `docs/plans/v4-owner-responses/gamepiece-score.md` | PENDING |
| `artifact_manifest_owner` | `docs/plans/v4-owner-responses/artifact-manifest.md` | PENDING |
| `evaluation_curriculum_owner` | `docs/plans/v4-owner-responses/evaluation-curriculum.md` | PENDING |

The delegated agents are technical decision owners for these scoped responses
under the user's explicit delegation instruction. Their decisions are not
human maintainer approvals, do not supersede repository governance, and do not
authorize implementation. No role may decide for another row or infer another
owner's consent.

## Required content in each response

Each response must be a standalone, reviewable Markdown artifact and include:

1. The exact owner role, delegated agent/task identity, response author, and
   response date.
2. The pinned V4 proposal and matrix commit IDs and full SHA-256 values from
   this plan.
3. A disposition of exactly one of:
   - **ADOPT** the exact pinned V4 proposal for this owner row;
   - **REJECT-WITH-REPLACEMENT**, with a versioned replacement artifact and
     full artifact hash plus the affected revised field table/payload bytes
     and cross-owner dependencies; or
   - **PENDING**, when the role cannot make either decision yet.
4. Evidence-based rationale with precise source/contract citations, the
   selected values or unresolved questions for this row, and explicit scope
   limits. A response may not silently edit the V4 proposal or treat a
   recommendation as normative.
5. A statement that the response records only this role's technical decision
   and does not claim human maintainer approval, cross-owner adoption, ABI
   publication, or implementation authorization.

An unsubstantiated rejection, a replacement without immutable version/hash,
or a response that changes another row is not a completed disposition and
must remain **PENDING**.

## Review and reconciliation gates

1. This plan's own Pass 1 review covers only the planning workflow. Its sole
   file allowlist is this document. A Pass 1 **ACCEPT** authorizes later
   documentation-only Pass 2 work on exactly the nine response paths listed
   above; it does not authorize any other file or action.
2. Each response is authored separately under its assigned role and receives
   an independent exact-head review. Until that review accepts the response,
   the corresponding matrix row remains **PENDING**. The author cannot review
   their own response.
3. A reviewed **ADOPT** or **REJECT-WITH-REPLACEMENT** is only that role's
   decision. Replacement artifacts trigger fresh cross-owner analysis against
   the exact replacement hash; affected response evidence must not be
   presumed to cover the replacement.
4. No shared ABI/profile is adopted by these nine responses alone. After all
   applicable rows have independently reviewed dispositions pinned to one
   compatible immutable artifact hash, a separate reconciliation artifact
   must resolve cross-owner fields, calculate and pin the final ABI/profile
   hash, and receive independent review. Any replacement or changed
   cross-dependent field requires a new immutable reconciliation version and
   recalculated payload/envelope sizes.
5. Each owner then needs a separate plan pinned to that adopted ABI/profile
   hash and its own independent review before schema-only work can be planned.
   Schema implementation, source changes, tests, compilation, runtime/device
   work, simulator/dashboard activity, checkpoint/campaign work, training,
   evaluation, and qualification each remain outside this plan and require
   separate reviewed scope and authorization.

At the time this plan is authored, **all nine dispositions and the shared ABI
publication gate are PENDING**. No owner adoption, normative ABI, source
authorization, or Stage 1 evidence is claimed.
