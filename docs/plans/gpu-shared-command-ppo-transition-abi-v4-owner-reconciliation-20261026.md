# GPU shared-command PPO transition ABI V4 owner reconciliation

Date: 2026-10-26
Status: **Reconciled proposal; implementation and publication are not authorized by this document**

## Purpose and authority

This document reconciles the nine independently reviewed owner responses for
the exact V4 proposal and adoption matrix listed below. All nine rows adopt
their row of the pinned proposal. The reconciliation is a technical proposal
record; it is not human maintainer approval, ABI publication, source
conformance, GPU-only compliance, or Stage 1 qualification.

The documentation-only closure plan at
`docs/plans/gpu-shared-command-ppo-transition-abi-v4-pending-owner-closure-plan-20261026.md`
authorizes this artifact only after the nine current rows have compatible,
independently reviewed dispositions. The exact response and review heads are
listed in the ledger below. No source, schema, runtime, tests, device,
dashboard, evaluation, campaign, or training changes are included.

| Input | Commit | SHA-256 |
|---|---|---|
| V4 proposal, `docs/plans/gpu-shared-command-ppo-transition-abi-v4-20261026.md` | `6b4ce4a939a2cdea375bd4be6879f6f7bc73141b` | `0c5ba39c3a4fa7c9bdf168ccf7c1dc792f4d3e27bd6f532517a967fca4b025b6` |
| V4 owner-adoption matrix, `docs/plans/gpu-shared-command-ppo-transition-abi-v4-owner-adoption-matrix-20261026.md` | `ee234b2c5162038e6fcfff6d9551d80f443f6d82` | `9888eb2a8d5e63b04aa328dca222dd08520a709e2ee78684f9d2dc07aafede53` |

## Exact owner responses and reviews

Every response below has disposition **ADOPT** for its own row. The response
SHA is over the exact file bytes. Review record SHA is over the exact persisted
Markdown record bytes at the listed path. Review ACCEPT is not a deployment or
governance approval.

| Owner row | Response commit | Response file SHA-256 | Disposition | Persisted exact-head review | Review record SHA-256 |
|---|---|---|---|---|---|
| Environment/profile | `cd9276c62edb695715bf54ef44dcd7703183ee5b` | `e0cac32ea061f73b96b57d378c35784f73643ea3b2dfb6fa3ca1a65e56d47a2a` | ADOPT | `docs/plans/v4-owner-responses/reviews/environment-profile.md` | `894de2c64d69f3ab9618b0780afea8fea72795977dbf18b1639bf45e8c9d8878` |
| Plan A | `03b7a9f5cbb161a2b30563e76a360619df144849` | `e6863a2b3b52fc2526062ac081cec0b153742d0f32344d8006d766cfad74df21` | ADOPT | `docs/plans/v4-owner-responses/reviews/plan-a.md` | `9a1eea2c57c1c20b59d6fdfd11e7ffb697aa77b0093f7fcb40a44978233f38e2` |
| Plan C | `6106e2d236af4a0891d8d72a578ce748aece5743` | `4bad45f3bb5dcbe87907f666805166364a9f424b2fd95a59b1dd23ef3c93c610` | ADOPT | `docs/plans/v4-owner-responses/reviews/plan-c.md` | `cb16ab85da0cd4778c5fe09042ccc3150ece64139a7ed76dee6d7e5b07c3f1a4` |
| Route | `499cd2bf5a4dead0975f4e14fcb6df25d7661858` | `f3e32aba59c9ba018e4e2d6a37bfee26b1ddb536d7e98ebf0ff1042aff481b82` | ADOPT | `docs/plans/v4-owner-responses/reviews/route.md` | `ac5e3aed336ca799e859db17e72478ef90b92027a1b043a83bf14d977a3a4a9b` |
| Observation/native | `f6af063a025eef967be442468355eb5a73d742c9` | `886a177f91250cdbbb4a9c4a767c3424a1b56a0ce0033c419a570facb26582d0` | ADOPT | `docs/plans/v4-owner-responses/reviews/observation-native.md` | `53649f76ee8204823cf7850cc3cbed7934a7b34a8a6fa71aec8210a906567cff` |
| PPO/RNG | `602565ef99986996d2c12d06024833247dc3d395` | `219f3466741c17d9e49a4d402f0b3f8ae6f016f0738a068034a8ad312f610d75` | ADOPT | `docs/plans/v4-owner-responses/reviews/ppo-rng.md` | `891efba9fd6b6a1a714085511c3d5f55ca660d8f3f598485f2ad165dd1cf9b76` |
| Gamepiece/score | `3738ad58fc883c5853b32c508e89ac657864bca4` | `f57476d13d6038e9e0109fedc557c617ead25a99e4db49512b135a532c1e13af` | ADOPT | `docs/plans/v4-owner-responses/reviews/gamepiece-score.md` | `8e58451551c82c4713688e3fd228a9b410125e2155157f437e75679b2e3601ac` |
| Artifact/manifest | `45dc7b4894d97fd7aa04b1f84aac3cdc207f0c9d` | `9b7a571bd98c6aed6700d445aea8e05a2308cdc5c1bcba9c93c961c4194d8289` | ADOPT | `docs/plans/v4-owner-responses/reviews/artifact-manifest.md` | `2fd229e4ba3832b3187b2f55960f05ae4fe8f3e09bccfd4fc77dd9d700ea5c2e` |
| Evaluation/curriculum | `57fe139b9f22f6bfee4bf874119117e26aa8f84c` | `4c4b06740f08b17396824626acee686d13d5ee3c0394538ec5449d244867d6d5` | ADOPT | `docs/plans/v4-owner-responses/reviews/evaluation-curriculum.md` | `edb065b320af7efdba75effcee447585bda4050464826bfb8527d5fe5d189c9e` |

The exact response heads and review files were re-read from their pushed
branches before this reconciliation. Earlier REVISE records remain in review
history where required; each row's current record pins the current response
head and ACCEPT verdict.

## Reconciled profile declaration

The following single-line UTF-8 JSON byte sequence, excluding the Markdown
fence and with no trailing newline, is the canonical profile/field-layout
declaration. Integers are decimal, arrays preserve the specified axis/order,
and JSON keys are in the exact order shown. Its SHA-256 is the V4
`profile_digest` and the final reconciled profile hash.

```json
{"abi":"FiveTrackPpoTransitionV4","envelope":"FiveTrackPpoBatchV4","envelope_version":2,"schema_version":4,"profile_id":"meco-five-track-shared-command-v4","track_codes":{"A1":1,"A2":2,"A3":3,"C1":4,"C2":5},"N":1,"S":6,"R":32,"T":12,"E":504,"K":16,"raster":{"shape":[6,25,51],"dtype":"float32","global_shape":[7],"global_order":["x","y","sin_heading","cos_heading","load","signed_hub_timer","match_remaining"]},"plan_a":{"nodes":12096,"support_rows":96768,"edges_per_node":10,"ancestor_levels":14,"cost_dtype":"float64"},"plan_c":{"source_rows":537,"raw_width":20,"model_width":40,"route_rows":9681,"route_chunks":152,"route_chunk_capacity":64,"path_points":73},"row_bytes":9822906,"rows_bytes":314332992,"score_event":{"slots":2304,"record_bytes":59,"table_bytes":135936},"static_identity_bytes":123,"payload_bytes":314469051,"header_bytes":96,"bundle_bytes":314469147,"encoding":"packed-little-endian-no-padding-float32-ieee754-valid-finite-mask-false-zero-signed-zero-preserved"}
```

### Capacity and field compatibility

| Area | Reconciled value and compatibility decision |
|---|---|
| Batch/profile | One track per run (`N=1`); `S=6` robot slots; `R=32` transitions; `T=12` fixed macro ticks; `E=504` observer target slots; `K=16` Plan C candidate selection plus fallback. Track codes are A1=1, A2=2, A3=3, C1=4, C2=5. |
| Observation | Float32 raster `[6,25,51]`; seven globals in the declared order. Observed targets use 504 slots and four-int handles. UUIDv4 environment instance and reset/sequence lifecycle follow the accepted Observation/native row. Both terminal and truncated may be true; terminal has precedence for macro status/bootstrap. |
| Plan A | Padded capacity 12,096 nodes, 96,768 ordered support rows, 10 graph edges per padded node, and 14 ancestor levels. `node_valid` selects the legal subset. Three V4 cost arrays use float64 as explicitly adopted; the proposal does not claim this is the current source tensor dtype. |
| Plan C | 537 source rows, 20 raw and 40 model features, K=16, stable source identity and explicit masks/ranks, ordinary and continuation maps, 9,681 route sink rows, 152 chunks of up to 64, and 73 path points. Rows 512–535 are ordered variant-major, four actions per variant; row 536 is continuation. Oracle rows retain their separate identity/sentinel domain. |
| Route | Route sink joins use the adopted source row and observation-handle contracts. All 9,681 sink rows, including inactive and oracle capacity, are serialized. P=73 path points; route map provenance and presence bits are bound in `support_digest`. Bump and learned costs are separately represented; route cost is 0.05 simulated score points/sec, risk cost is capped at 0.75 simulated score points within 2 m using the adopted forecast weighting. |
| PPO/RNG | Transition validity, action linkage, terminal/truncation precedence, update atomicity and bootstrap semantics follow the adopted row. Separate PPO stream/site registries preserve V4 bytes; stream codes are 1 action-categorical and 2 epoch-permutation; draw-site codes are 1 categorical-word and 2 Fisher-Yates-word. RNG roots/counters/draw order and failure rules are source-bound. |
| Gamepiece/score | Fixed 2,304 × 59-byte typed score-event table. Full capacity is reserved; invalid slots have valid=0 and other fields zero. Event identity and two source-site codes are adopted; overflow or invalid event fails the whole transition before mutation. |
| Artifact/manifest | Envelope v2 uses the 96-byte little-endian header, 123-byte static identity prefix, exact payload size 314,469,051, and exact full bundle size 314,469,147. Full fixed-capacity event table; source/config/checkpoint digest spans and strict version-first exact-length fail-closed reader predicate are adopted. |
| Evaluation/curriculum | User selected the limited `simulated_match_score`: accepted active-HUB FUEL awards only, not official match points. User selected fixed strategic-horizon gross reward with route/risk costs separate and confirmed current counterfactual defaults: 160 physics ticks, at most 12 ticks per macro (last may be partial), gamma 0.993 per macro with first branch exponent 0 and one increment per continuation decision. Route and risk costs retain their separate units above. |

The canonical declaration hash is computed over only the exact JSON content
inside the fenced block, with no leading/trailing whitespace or newline. Its
SHA-256, which is the final reconciled profile hash, is
`25c03d281f8866714bc8b58308373de827c0e6fb926d5f498497c9af685cb476`. The
review record pins this file's immutable SHA-256 and this declaration hash.

## Byte derivation

The packed row subtotal is derived from the adopted V4 field inventory. No
implicit alignment, struct padding, or trailing bytes are included.

| Per-transition field group | Bytes |
|---|---:|
| Identity/schema, six int64, two uint64, environment UUID pair, eleven bools | 94 |
| Observation raster `[6,25,51]` float32 plus seven float32 globals | 30,628 |
| Recurrent hidden before/after, each float32[64] | 512 |
| Plan C source/model feature tables, masks, IDs, maps and ranks | 149,959 |
| Observer target table `[504]`, XY/velocity/age/confidence/valid/handle | 28,728 |
| Plan A support/search arrays, including 3×float64[12,096], node/edge/jump evidence | 1,983,764 |
| Plan C full route sink: 9,681 × 788 bytes/query | 7,628,628 |
| Common selected command | 99 |
| PPO reward/log-probability/entropy/value/old value/return/advantage | 28 |
| Two RNG streams, 40 bytes each | 80 |
| Input/support/transition digests and replay identifiers | 122 |
| Fixed 12-tick macro trace | 264 |
| **Bytes per transition** | **9,822,906** |

`9,822,906 × 32 = 314,332,992` transition bytes. The score table bound is
`R × S × T = 32 × 6 × 12 = 2,304` records; each record is 59 bytes, so
`2,304 × 59 = 135,936` bytes. Static source/config/checkpoint digests and RNG
identity total 123 bytes. Thus payload is
`314,332,992 + 135,936 + 123 = 314,469,051` bytes. The 96-byte envelope makes
the complete bundle `314,469,051 + 96 = 314,469,147` bytes (299.9012 MiB).

The arithmetic is independently checked against the V4 owner response and
review. The declaration's `row_bytes`, `rows_bytes`, `score_event.table_bytes`,
`payload_bytes`, and `bundle_bytes` equal these derived sums.

## Implementation and qualification gates that remain

The reconciled contract specifies the target; it does not claim current source
conformance. The reviewed Evaluation/curriculum row records that the existing
PPO path does not yet implement the selected 160-tick on-policy target, uses a
single `done=ended or timeout` flag for GAE/bootstrap, and the counterfactual
collector does not yet type its end cause. The accepted ABI semantics preserve
terminal and truncation separately and define terminal precedence. These source
gaps require a separate implementation plan against this exact profile hash.

The user-approved simulated-score choice still needs its immutable
source/rules-bound Phase 0 artifact and digest binding. Before qualification,
the frozen Phase 0 calibration, variance/margin evidence, target-scoped v5
protocol, five checkpoint manifests, and three consecutive passing Stage 1
rounds per track must be completed. Each track must meet the agreed category,
layout, D0 safety, paired D1/D2, and score/progress gates with bound artifacts.
No current diagnostic training chunk, earlier 100/100 episode report, or
response/review is qualification evidence.

This reconciliation neither publishes V4 as normative nor implies governance
approval. The next authorized step is a separate implementation plan pinned
to this canonical profile hash, reviewed under Pass 1, then scoped source/schema
implementation with exact-head reviews. Simulator behavior changes must follow
the project AGENTS.md requirement for a representative seeded end-to-end
simulator evaluation and a current dashboard/replay check. GPU-only scope also
requires a reviewed, measured residency plan; this reconciliation makes no
claim that training, simulation, scoring, calibration, or replay is fully
GPU-resident.
