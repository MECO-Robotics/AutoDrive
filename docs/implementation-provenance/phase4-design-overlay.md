# Phase 4 validation design overlay

## Three-pass record

- **Planner:** audited the Phase 4 requirements against the existing scenario
  and validation protocol. The plan defines 11 requirement IDs, exact matched
  treatment/control case axes, validation-only seed inheritance, whole-family
  separation, and distinct design-only axes. It specifies no final-test seeds,
  safety margins, or qualification claim.
- **Implementer:** `69e7e79b61113e9b64e90ce8e676a60b3b25b777` compiles the
  protocol-bound registry into an overlay. Follow-up `4033acc8a2a67f0d206247363e6c98ca21ce7a97`
  adds exact route-segment selection for trench interception (14 registered
  trench-route categories), defines robot-to-robot contact separately from
  field-bump terrain, and makes the contact/recovery event definitions
  explicit without inventing thresholds. The integration candidate contains
  the contract at `5a37fc73c509c3c785958d279b3845cc2b6eb262` and the runtime modules at
  `f2867715c08e71fd6365b8ddb5ba742e62109fba`.
- **Independent reviewer:** approved exact `69e7e79b61113e9b64e90ce8e676a60b3b25b777`
  and follow-up `4033acc8a2a67f0d206247363e6c98ca21ce7a97`, checking selector
  coverage, contact semantics, overlay digests, and absence of final-test
  seed values/rows.

## Evidence and limits

The checked-in overlay binds protocol
`dec8964454769b01fa4d79fe5f1cb7919013d94b5076e3a99a20ba5a4181f6f2`, its raw
file SHA-256, and the registry digest. It contains 466 unique validation case
axes and no serialized seed values or final-test rows. The exact coverage is
factor-specific: pressure factors resolve to 152 case axes each; the reactive
pool to 304; starts to 152; teammate protocol cases to 188; sensor factors to
466; bump cases to 304; and pose mislocalization to 152. Separate design-only
axes contain 7 opponent policies, 3 legal team-pose sets, 5 teammate profiles,
3 bump geometries, and 3 focal-pose offsets. These design-only rows are not
counted as protocol or executed coverage.

The original contract implementation had five focused tests; the scope fix had
eight focused tests. Both report Python compilation, actual protocol overlay
generation, and `git diff --check`. No simulator or GPU was used. The design
contract now precisely scopes planned factors, while runtime stimuli are still
not wired to the curriculum/evaluation runner. Full-factor campaign execution,
simulator replays, realized outcome signals, qualification, and deployment
evidence remain outstanding. The current training source remains separate and
unchanged.
