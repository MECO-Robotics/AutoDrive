# Phase 4 runtime evidence capture

## Planner pass

The reviewed design registry has eleven requirements, but campaign rows were
being marked executable based only on design-axis shape. The next contract
tranche binds each requirement to frozen factor IDs, available runtime events,
and required outcome signals. The campaign must remain non-executable unless
the factor stimulus and every required signal have authoritative sources.
Existing supported factor configs should record realized stimuli without
claiming that unavailable outcome metrics exist. Final-test seeds must not be
serialized into this capture. No simulator/GPU/replay work is part of this
tranche while the continuous main-source training run remains active.

## Implementer pass

Implementation `f9e53044d01688069dea54e2ea2d5438b76fafc8` adds the binding for
all eleven registry requirements and source-bound episode captures for the
four already registered runtime factors: feint reversal, abrupt policy
switch, observation delay, and low confidence. Matrix intake validates the
factor and digest, evaluator source hash, split-specific seed behavior,
scenario/category, qualification case, realized runtime event, and exact
required/observed/unavailable signal accounting. The campaign compiler marks
a row executable only if both its runtime stimulus and all required signals
are supported; the executable qualification-cell count is zero. This tranche
does not wire the remaining opponent/context stimuli or create missing
outcome signals.

## Independent reviewer pass

Reviewer approved exact implementer head
`f9e53044d01688069dea54e2ea2d5438b76fafc8` after requesting and receiving
fail-closed fixes for unrealized events, episode-seed/case binding, and strict
capture validation. The reviewer reran 33 focused contract, campaign-plan,
and matrix tests. No simulator, GPU, training, replay, or qualification
evidence was generated.

## Limits

The four existing factors can produce diagnostic realized-stimulus records,
but their registry-required outcome signals remain unavailable. Stationary
blockage, direct pursuit, trench interception, reactive opponent pools, legal
team starts, teammate behaviors, contact/recovery, and pose offsets still
lack validation runtime wiring. Contact proxies are not treated as required
outcomes. This code tranche does not make any Phase 4 cell qualification
executable and is not simulator validated.
