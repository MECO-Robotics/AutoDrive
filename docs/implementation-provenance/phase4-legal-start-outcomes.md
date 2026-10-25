# Phase 4 legal start profile outcome capture

## Planner pass

The planner identified that full-team start-profile assignment and paired dispatch
already existed, but runtime outcome fields were null and the reducer validated
coverage only. The tranche closes that reporting gap while retaining the
diagnostic-only qualification state.

## Implementer pass

Implementation commits: `2edfdb73d1f7dac61d69ece982605c3ce196c4f5` and
`6711e0aa1d8b4cd0a98bf6e4019e9979432fc557`.

After applying a profile and before constructing the observation, runtime capture
records legality for every simulator world from the poses actually applied. The
paired reducer requires exact policy/category/seed coverage, valid per-world
legality, and finite matching outcomes. It reports matched per-seed deltas and a
coverage digest that binds legality vectors, world counts, rates, assignments,
source/checkpoint/device identity, and outcome data. It rejects runtime profile
names that differ from the frozen assignment. Assignment objects remain
immutable, and qualification remains disabled.

Focused contract evidence: the initial focused Phase 4 suites reported 41 passed;
after the final follow-up, the dispatch contract reported 1 passed and the runtime
capture cases reported 3 passed. Python compilation and `git diff --check` passed.

## Reviewer pass

Independent exact-head review approved `6711e0aa1d8b4cd0a98bf6e4019e9979432fc557`
for validation wiring. The review confirmed exact pairing, finite metric checks,
runtime profile binding, and coverage digest inclusion of legality evidence.
The reviewed dispatch invokes the legal-start treatment without a later
stationary-blockage reset modifier: it supplies `--phase4-start-profile` and no
`--phase4-factor`; the environment also rejects simultaneous contact-geometry
and full-team start-profile assignments.

## Limits

This is code and contract evidence, not simulator validation. No seeded Phase 4
paired evaluation, replay inspection, GPU run, or current-dashboard replay check
was performed. A seeded `full_match_offense` run and fresh replay review after
restarting and verifying the dashboard are still required. Qualification remains
false.
