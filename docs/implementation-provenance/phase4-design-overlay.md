# Phase 4 validation design overlay

This tranche compiles an explicit validation-only design for all eleven
Phase 4 requirements in the source plan. Run it with:

```bash
python3 -m frc_defense.meco_phase4_design \
  --protocol qualifications/meco/protocol.json \
  --registry qualifications/meco/phase4_design_registry.json \
  --output qualifications/meco/phase4_design_overlay.json
```

The output binds the qualification protocol ID and raw file SHA-256, plus the
raw factor-registry SHA-256. It resolves each selector to exact factor-specific
scenario, category, layout, defender-family, start, and sensor case rows. It
reports planned matched factor/no-factor coverage, factor IDs, required outcome
signals, and explicit availability states. Design-only axes such as opponent
policy, team-pose, teammate, bump-geometry, and pose-offset variants are emitted
as separate cross-product rows and are explicitly excluded from protocol and
executed coverage counts. Unknown selectors fail closed. The compiler also
checks the exact eleven source-plan requirement IDs.

The compiler verifies whole defender-policy-family separation for training and
validation; the no-defender `none` case is not treated as a policy family. The
overlay does not serialize seed values or final-test rows.

This is a design/compiler contract only. The currently registered runtime
factors cover feint reversal, 120 ms observation delay, independent low
confidence, and abrupt orbit-to-pursuit switching. Other requirements are
marked design-only. No new simulator stimuli are implemented here, and no
factor outcomes were run. Generic score/contact telemetry is identified only
as a proxy where applicable; factor-specific reaction, interception, crossing,
recovery, and teammate-behavior signals remain unavailable. No safety margins
or qualification claim are included.

The checked-in overlay is bound to protocol `dec8964454769b01fa4d79fe5f1cb7919013d94b5076e3a99a20ba5a4181f6f2`
and contains an inventory of 466 unique validation case-axis rows. Factor-scoped
protocol case counts are 152 for each pressure-category condition, 304 for the
reactive defender pool across the two pressure scenarios, 152 for full-match
team starts, 188 across full-match and team-delivery teammate cases, 466 for
sensor conditions, 304 for pressure/full-match bump cases, and 152 for
full-match pose mislocalization. These counts describe planned protocol axes,
not executed episodes; design-only axes remain separate.
