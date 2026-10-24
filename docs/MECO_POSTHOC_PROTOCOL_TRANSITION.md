# MECO post-hoc protocol transition

`scripts/meco_protocol_transition.py` audits a completed scheduler segment and
can export a matched five-track actor-only initialization artifact. The segment
is always labeled `unbound_nonqualifying_historical_segment`; retrospective
train-seed membership does not qualify earlier training or evaluation.

The audit checks that there is no pending scheduler assignment, scheduler
cycles and lane outcomes reach one common update frontier, every task seed is
in the supplied frozen protocol's exact train case, all five run manifests and
checkpoints agree on full source identity/configuration/protocol/run IDs, and
per-update history assignment IDs agree with the scheduler. Optional evaluation
reports are checked against the full source identity and exact five checkpoint
hashes. The report stores hashes of the scheduler, checkpoints, and evaluations.

```bash
python3 scripts/meco_protocol_transition.py audit \
  --segment-dir /path/to/old-supervisor-output \
  --qualification-protocol /path/to/frozen-qualification-protocol.json \
  --evaluation /path/to/matched-evaluation.json \
  --output /path/to/segment-audit.json

python3 scripts/meco_protocol_transition.py export \
  --audit /path/to/segment-audit.json \
  --output /path/to/weights-only-transition.pt
```

Export rechecks checkpoint byte hashes and actor-state hashes, constructs each
registered track model on CPU, and strictly validates its actor state dict.
The artifact and JSON sidecar bind all five checkpoint identities, model
shapes/dtypes, source lineage, checked protocol identity, evaluation hashes,
and the audit hash. The export is not a trainer resume checkpoint.

For a later explicit initialization path, Python callers may use
`consume_transition(path, seed=...)`. It loads only the five actors, creates
new value heads and empty Adam optimizers, seeds Python and PyTorch from the
required fresh seed, issues a new run ID, and returns update/episode zero with
empty history. No old critic, optimizer, random state, run ID, or counters are
read. This helper is not wired into the supervisor or `--resume`; integration
must be reviewed separately. Existing resume identity checks are unchanged.

The audit is a consistency check over stored artifacts. It cannot prove that
the original process followed its declared protocol, recover absent history,
or establish simulator performance. The transition does not make the old
segment qualification evidence and does not by itself establish promotion.
