# Qualification v4 matched-reference protocol

## Planner

The candidate had only a provisional v1 protocol and D0 calibration for
`pickup_simple`; it lacked repeated matched-reference calibration, explicit
reviewed margins, and a protocol binding those inputs. The planner identified
the necessary v4 contract: match D1/D2 reference rows to frozen D0 case/seed
tables; bind source, runtime, predictor mode, candidate settings, repeats,
safety rows, and hashes; keep operational margin choice explicit and reviewed;
and admit new promotion claims only after the v4 protocol passes validation.
The existing `pickup_simple` D0 report is not generalized to other target
closures.

## Implementer

Implemented on an isolated worktree from candidate base
`78d230cdf44bb502c1f69958a21673bffc89435c`, then integrated as
`68997963a76226c1f1dad0a1021d3a4432b98696`.
The implementation adds matched-reference report and bundle intake, immutable
v4 protocol construction/validation, exact repeat/seed key checks, and
calibration-only D1/D2 evaluator bindings. Follow-up commits
`568ce4c3d52b4f752fc67a6807957ec352aee053` and
`73d50d01b52efa15534fc8960529bfa4ce0f0914` wire v4 through protocol loading,
dashboard status, manifests, campaign planning, and the validation runner.
New qualification claims require v4 with the exact reviewed-margin gate;
v1-v3 remain readable for diagnostic or legacy behavior only. Campaign
artifacts remain validation-only and do not themselves qualify a policy.

## Reviewer

The independent reviewer requested changes on the initial v4 and first
workflow-integration commits, identifying held-out loader, dashboard status,
campaign admission, and legacy-eligibility gaps. Those findings were fixed.
The reviewer approved v4 policy commit
`73d50d01b52efa15534fc8960529bfa4ce0f0914` and then approved the combined
candidate at exact head `68997963a76226c1f1dad0a1021d3a4432b98696`.

## Validation and limits

The combined candidate suite passed 128 scoped tests in 164.87 seconds.
Scoped implementation checks also reported compilation and diff checks. No
D0/D1/D2 runtime calibration was run, no operational margin artifact was
selected or approved, and no v4 protocol or qualification result was
produced. The checked-in protocol remains provisional v1; only a D0 report
for `pickup_simple` exists. No simulator replay, training, Hailo profiling,
or deployment qualification is claimed. Raw agent transcripts were not
persisted; this file summarizes the actual planner, implementer, and reviewer
reports.
