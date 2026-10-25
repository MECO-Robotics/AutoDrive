# Plan A human choice authority v1

The Ed25519 verifier in `frc_defense/meco_plan_a_expert_choice.py` is wired
through the Phase 2 preference-v2 materializer and fitter. Signatures bind the
annotator identity and basis to the canonical sidecar, source, and replayed
legal-node content hashes. The fitter's fresh replay loader verifies the
qualification-protocol-pinned key, joins exact decision identities, checks the
available selected pair, and independently recomputes the minimum-route
baseline before accepting a different human choice. It trains a pairwise
preference over complete actions scored by U+B. The existing minimum-route
producer remains a baseline and is not labeled human expert.

Focused CPU tests exercise the actual materializer, fresh replay loader, fitter,
and weights-only artifact validator with synthetic data, generated keys, a
mocked deterministic replay adapter, and a small CPU test model. No real source
collection, simulator execution/replay, GPU operation, or dashboard check was
performed. Production use still requires a qualification protocol containing
an independently controlled pinned public-key registry and real replay
evidence.

## Three-pass review record

Planner audit, implementation, and independent exact-head review were recorded
as a scoped three-pass tranche. Implementer commits are
`066cc0b02` (authority contract), `a939e406a` (materializer/loader/fitter),
`08c6a0480` (keep complete-action choices joint), and `6e6ef8a5f`
(bind fitted pair targets to signed authority). Current-main integration
`c118f0db07c723944389ebcca788923cbb439535` was independently reviewed and
matches the final reviewed feature code exactly.

The follow-up rejects JSONL target rewrites even when a valid signed sidecar is
retained. It also ensures a complete-action preference does not create an
unsupported U-only goal ranking. The focused suite reported 19 passing CPU
contract tests. Replay uses a mocked adapter and fitting uses a small CPU test
model; real expert annotation, production replay, GPU fitting, and campaign
warm-start remain unverified.
