# Plan C repeated-label corpus coverage audit

## Planner pass

The repeated-rollout collector and fitter validate individual labels, but
there was no corpus-level report showing how much stable preference signal a
collected Plan C train dataset contains. Plan sections 5, 6, and 7 call for
matched-rollout variability handling, strategic regret/calibration, and
candidate recall diagnostics. Candidate coverage must remain explicitly
bounded to the sampled candidate set; it cannot establish full-action recall.

## Implementer pass

Commit `3785019824db8a1ff0b139390a5f22dc84519048` adds an offline audit module
and CLI that reuse the strict source-bound counterfactual dataset reader. The
report summarizes train-split prefixes, sampled candidates, pair labels,
skipped branches, and category/action coverage, and binds deterministic report
identity to dataset, manifest, source, and protocol hashes. It does not infer
minimum adequacy thresholds or claim warm-start readiness.

Fix commit `dbf1da3bb8801eca5d243a1521990299a0caedd7` adds an audit-boundary
check that row seeds exactly cover the manifest-declared train seeds and makes
repeated-label boolean typing exact. Regression cases cover an omitted seed
and integer `1` in place of a boolean. The focused Plan C audit/repeated-label
suite passed 22 tests; Python compilation, CLI help, and diff checks passed.

Integrated code commits are `70397a016` and `5389e618a`.

## Independent reviewer pass

Independent exact-head review approved `dbf1da3bb8801eca5d243a1521990299a0caedd7`.
It confirmed source, protocol, dataset hash, train split, complete repeated
labels, exact declared-seed coverage, deterministic identity, and descriptive
limits. No game-behavior simulator or device validation applies to this offline
dataset audit; none was performed.

## Evidence limits

No real repeated-label dataset was audited, and this work creates no labels or
weights. It does not establish dataset adequacy, candidate recall against the
full action space, value calibration, warm-start readiness, simulator
performance, or qualification. No training process or GPU lease was changed.
