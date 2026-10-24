# Phase 4 pair-contact event summary

## Planner pass

The signal audit found that the simulator exposed per-robot contact flags but
discarded pair IDs and impulses in both the Torch and fused six-robot HIP
solvers. The bounded implementation captures the registered pair (2,3) from
the solver paths, with end-of-substep timestamps and impulse summaries. The
plan does not treat this as recovery or route-completion evidence, and it does
not claim a full event stream. Validation geometry remains unscheduled.

## Implementer pass

Implementation began at `42e8e662e6f3b6f9ca167d7d4b40e24be55578b4` and was
corrected at `b3adb19b055acece92952c69bed7c1e9e1290699`. Candidate integration
commits are `0d1e5eabf649bb6eafbd50f83f5093cf9ff6c6b2` and
`62b4a00dd0eb2c7646b478c932e651d33537dab2`. Torch and both HIP launch paths
deduplicate contact counts by physical substep while separately counting
solver reports; they sum normal, signed tangential, and net impulse across
solver passes. The episode record binds the summary to `contact.robot_bump`
and slots (2,3). The summary preserves first and last completed-substep
timestamps, not a full event chronology.

## Independent reviewer pass

The reviewer first requested a timestamp correction because the initial
version recorded the pre-step clock. The corrected implementation records
substep index plus one and time plus `dt` in Torch and both HIP paths. The
reviewer approved the exact corrected working diff, which was committed as
`b3adb19b055acece92952c69bed7c1e9e1290699`.

## Checks and limits

Seventeen focused static/contract tests passed. Python compilation and
`git diff --check` passed. No simulator behavior test, GPU test, HIP extension
compile, or replay was run. Contact geometry is not scheduled by validation;
recovery and route-completion signals remain unavailable. The dashboard pin
was updated with this manifest entry, but the dashboard service has not been
restarted against this candidate state.
