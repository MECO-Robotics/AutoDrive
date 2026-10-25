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
