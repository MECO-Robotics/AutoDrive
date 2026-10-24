# Phase 4 validation design overlay

## Three-pass record

- **Planner:** audited the Phase 4 requirements against the existing scenario
  and validation protocol. The plan defines 11 requirement IDs, exact matched
  treatment/control case axes, validation-only seed inheritance, whole-family
  separation, and distinct design-only axes. It specifies no final-test seeds,
  safety margins, or qualification claim.
- **Implementer:** `69e7e79b61113e9b64e90ce8e676a60b3b25b777` compiles the
  protocol-bound registry into an overlay. It resolves selectors to exact
  protocol axes, enforces all 11 IDs, rejects unknown selectors, and emits
  design-only policy/team-pose/teammate/bump/pose variants separately from
  protocol and executed coverage. The integration candidate contains this
  change at `4375a174d1866f2c2cf6ba8d8f994f5bde082328`.
- **Independent reviewer:** approved exact implementer commit
  `69e7e79b61113e9b64e90ce8e676a60b3b25b777`, specifically verifying the
  corrected selector resolution and factor-specific coverage scopes.

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

The implementer reports five focused tests passing, plus Python compilation,
actual protocol compilation, and `git diff --check`. No simulator or GPU was
used. This tranche supplies a reviewed design/compiler contract only: runtime
stimuli, full-factor campaign execution, simulator replays, realized outcome
signals, qualification, and deployment evidence remain outstanding. The
current training source remains separate and unchanged.
