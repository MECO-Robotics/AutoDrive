# Paired matrix effects and confidence intervals: three-pass summary

## Planner pass

Planner reference: followup task `/root/phase4_next_planner`. It audited the
evaluation plan's matched-comparison requirements and the frozen matrix output. Existing
matrix rows preserved matched outcomes, but the postprocessor did not provide
paired-effect confidence intervals. The scoped recommendation was to summarize
all five learned tracks against D0, include the producer's named contrasts
where supported, use explicit equal category weights for a stratified pooled
estimate, and fail closed on duplicate, missing, or nonfinite paired data.
Bootstrap randomness must be deterministic and bound to the exact matrix and
condition-report bytes. The planner report was provided in the scoped agent
message; no separate original artifact was persisted.

## Implementer pass

The exact reviewed implementation commit is
`185aba4b411eed9698df0e251c469dfc580bb141` (`Add paired matrix confidence
intervals`), integrated in candidate commit
`3d5f4b0df3ff97795c8733d743040cb23d14c3bb`. It changes
`scripts/summarize_meco_ablation_matrix.py` and
`tests/test_summarize_meco_ablation_matrix.py`.

The CPU-only postprocessor retains per-category paired differences and adds an
equal-category-weighted effect with within-category paired-episode percentile
bootstrap intervals. Its seed is bound to the caller seed and exact matrix and
evaluation-report hashes. It validates the required five learned-track/D0
contrasts and the exact named pair catalog, rejects duplicate or incomplete
pairs and missing or nonfinite metrics, preserves `cv_minus_current` direction,
and reports null paired metrics as unavailable with an explicit reason. Pooled
input identity retains each matrix/condition/report hash even when condition
IDs repeat. The output remains diagnostic-only and does not infer promotion or
qualification.

## Independent reviewer pass

The independent reviewer audited the exact candidate diff and required
removing any complete expected contrast to invalidate a rehashed matrix. The
implementation now requires all five learned-track/D0 comparisons and every
named pair from `POLICY_CONTRASTS`, including all three A1/A2/A3 comparisons;
unsupported named pairs fail closed. The review also required explicit
unavailable handling for both-null metrics and per-matrix retention in the
pooled input hash. The exact reviewed commit
`185aba4b411eed9698df0e251c469dfc580bb141` was approved and integrated at
`3d5f4b0df3ff97795c8733d743040cb23d14c3bb`. Reviewer reference: final approval
from scoped reviewer task `/root/phase4_pair_contact_review`.

## Checks and limits

The focused summarizer suite passed 39 tests. Python compilation and
`git diff --check` passed. No simulator, GPU, training, dashboard replay, or
campaign was run for this tranche. The analyzer summarizes existing matrix
reports only; its intervals are diagnostic and cannot establish a plan gate,
promotion, qualification, or deployment readiness. The wider evaluation plan
remains incomplete.
