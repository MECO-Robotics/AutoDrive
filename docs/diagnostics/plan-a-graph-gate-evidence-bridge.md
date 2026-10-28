# Plan A graph gate evidence bridge

The closed-gate qualification was collected at `b72990fdf08b4f5c1cb8ce29c4b781f7c414283b`. Both physically attested WX9100 devices passed all fixture cases. The current integration has the same pinned route source and fixture hashes; the integration commit and generated HIP file state differ from the clean qualification checkout.

`scripts/verify_plan_a_graph_gate_evidence.py` checks reviewed report hashes, candidate commit source contents, fixture identity, graph result fields and support digests, and the two device identities. It permits exactly the graph gate literal to change when comparing pinned code. It requires the preweighted gate to stay closed and the current fixture manifest to be rebound to the current pinned sources. It performs no dispatch and cannot authenticate a report's author: the supplied report hashes must come from a trusted review of the retained original reports. It does not establish simulator, policy, optimizer, or training behavior.

For the retained evidence:

```sh
PYTHONPATH=. python scripts/verify_plan_a_graph_gate_evidence.py \
  --report /home/brian/Projects/.worktrees/autodrive-scoped-gate-evidence-20261028/b729-cuda0/plan-a-hip-evidence-20261028T224536.177961Z-4405a32d53a6.json \
  --report-sha256 39655e686137519f5a0b3bf7c5133ec4faf7673b0218e46dcbdd5e494bc79901 \
  --report /home/brian/Projects/.worktrees/autodrive-scoped-gate-evidence-20261028/b729-cuda1/plan-a-hip-evidence-20261028T224635.201073Z-d91b2457c0eb.json \
  --report-sha256 6c53e4a63e7a393b9075994dd9a5043aed1cd38a62d2fd30e53195fee61c38d2
```

A separately reviewed graph-only gate change still needs a refreshed source-bound fixture manifest, a test of actual gated dispatch, and a seeded A-track training rollout and simulator replay under an exclusive device lease. The current closed-gate harness refuses to run after either gate opens, so its reports must be retained as evidence of the pre-change candidate. The preweighted entry remains independently gated.

The fixture profile's graph CPU timing rebuilds edge weights for each call. It is not a cached-CPU comparison or a training speedup. On the same pickup_simple graph, separate probes measured the cached CPU oracle at about 20 ms and adaptive HIP at about 18 ms. Those measurements were not paired and exclude graph construction, transfers, policy, simulation, PPO, and compilation.

The A-track trainer selects the graph route on the indexed accelerator, but it presently materializes a compact selected-action payload for host-controlled macro dispatch, reads scalar and audit values, and converts the PPO permutation to a host list. Model, route and simulator tensor placement must be verified on an actual A rollout. A strict requirement of zero device-to-host decisions would require additional changes to dispatch and PPO control; this evidence does not establish that stronger property.
