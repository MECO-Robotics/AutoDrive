# Phase 4 validation campaign plan compiler

`frc_defense.meco_phase4_campaign_plan` compiles a compact, validation-only
schedule from the frozen qualification protocol, Phase 4 factor registry, and
approved design overlay. The compiler verifies protocol and overlay IDs,
raw-file digests, exact case inventory, validation seed assignment digests,
the 14 trench category declaration, and the complete D0/D1/D2/A1/A2/A3/C1/C2
control/treatment grid. It does not include seed arrays or final-test rows.

The plan describes 6,488 factor-by-case-by-design-axis cells and a theoretical
10,380,800 paired episode rows (100 validation seeds × eight policies × two
conditions per cell). It stores compact cell descriptors and counts, not the
episode rows. Design-axis cells are schedulable planning entries only and
remain non-executable until their simulator stimuli and evaluator handling are
implemented and verified. This artifact is not an evaluation, replay,
qualification result, or promotion approval.

Runtime/evaluator wiring still required includes: applying each runtime factor
to its intended validation case; implementing and verifying the policy-pool,
team-pose, teammate, contact, and focal-pose design stimuli; evaluating exact
matched policy/checkpoint and no-factor/treatment episodes; persisting
replay-linked per-episode records; enforcing resumable execution, exact pair
coverage, and interruption recovery; and computing the registered outcome
signals and uncertainty summaries. Any simulator behavior validation must
also be accompanied by a fresh end-to-end simulator replay under a verified
current dashboard stack, per project instructions.

No simulator, GPU, replay, dashboard, or live training process is used by this
compiler.
