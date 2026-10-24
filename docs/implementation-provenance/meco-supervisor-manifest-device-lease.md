# Manifest-driven supervisor and shared device leases

## Planner pass

The gap audit found the offline five-track manifest could not launch because
the supervisor was pinned to legacy output roots. It also found evaluation
and training device exclusion was not shared across processes. The scoped
implementation adds explicit manifest mode to the supervisor, derives track
configurations and output roots from the validated manifest, rejects runtime
overrides, and requires a deliberate diagnostic-launch flag. It adds stable,
sorted, exclusive leases for indexed devices and shares them with the
validation campaign runner. It does not collect data, start training, or
qualify policies.

## Implementer pass

The implementation is commit
`5d81713310e9400b12fead981f118913f83222f0`; candidate integration commit is
`55e988596`. The supervisor persists manifest digest, campaign identity and
roots, source/config/protocol bindings, lease contract, and hard-false Phase 2
completion and qualification fields. Manifest launches claim both fresh output
roots exclusively after acquiring sorted device leases. If evaluation-root
claiming fails, the newly created campaign root is removed only when it is
still empty. Validation-runner output creation now occurs after lease
acquisition. The live legacy run was not changed and acquires no retroactive
lease.

The scoped offline suite passed **35 tests** across device leases, supervisor
dispatch, validation campaign runner, and training-manifest contracts.
Compilation and `git diff --check` passed. No simulator, GPU, training, or
replay was run.

## Independent reviewer pass

The first review rejected a race between construction-time fresh-root checks
and launch. The implementation added post-lease exclusive root claims and
regression tests for roots appearing after construction, an evaluation root
appearing during claim, cleanup of an empty partial claim, and lease release
after rejection. The reviewer approved the corrected exact snapshot and
confirmed both roots cannot be reused in manifest mode. The reviewer did not
rerun the tests.

## Limits and next evidence

This closes manifest-to-supervisor launch wiring and cross-process device
coordination. It does not complete Phase 2. No protocol-bound counterfactual
dataset or warm-start weights exist for C1/C2; Plan A still needs U/B labels
and independent P-route labels. `phase2_complete` and
`qualification_eligible` remain false. The current live trainer predates this
lease contract and remains a legacy diagnostic campaign. A fresh simulator
evaluation and replay are still required before claiming simulation behavior
or validating dashboard replay provenance.
