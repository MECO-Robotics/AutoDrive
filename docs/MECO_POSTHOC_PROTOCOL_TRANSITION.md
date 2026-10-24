# MECO post-hoc protocol transition

`scripts/meco_protocol_transition.py` audits a completed scheduler segment and
can export a matched five-track actor-only initialization artifact. The segment
is always labeled `unbound_nonqualifying_historical_segment`; retrospective
train-seed membership does not qualify earlier training or evaluation.

The audit checks that there is no pending scheduler assignment, scheduler
cycles and lane outcomes reach one common update frontier, every task seed is
in the supplied frozen protocol's exact train case, all five run manifests and
checkpoints agree on full source identity/configuration/protocol/run IDs, and
per-update history assignment IDs agree with the scheduler. The complete
evaluation root is mandatory: every direct report JSON must have a completed
matching status sidecar, and a supervisor-written append-only,
hash-chained `evaluation-history.json` must enumerate every terminal evaluation.
Queue status anchors the history count and chain head. Each entry binds report
and status hashes plus the complete five-checkpoint snapshot inventory. Missing
history in a legacy run makes completeness unverifiable, so audit rejects it
instead of claiming the remaining files are complete. Reports with
absent qualification provenance, unknown cases, train-split evaluations, or
validation/test seed leakage fail closed. Existing automatic evaluations
without a qualification protocol ID are rejected for transition; they cannot
be omitted by passing a selected subset. The report stores hashes of the
scheduler, checkpoints, evaluation reports, status sidecars, and root inventory.

```bash
python3 scripts/meco_protocol_transition.py audit \
  --segment-dir /path/to/old-supervisor-output \
  --qualification-protocol /path/to/frozen-qualification-protocol.json \
  --evaluation-root /path/to/supervisor-evaluation-output \
  --output /path/to/segment-audit.json

python3 scripts/meco_protocol_transition.py export \
  --audit /path/to/segment-audit.json \
  --output /path/to/weights-only-transition.pt
```

Export rechecks checkpoint byte hashes and actor-state hashes, the history chain,
every report/status hash, and every checkpoint snapshot path/hash and complete
snapshot directory inventory. It constructs each
registered track model on CPU, and strictly validates its actor state dict.
The artifact and JSON sidecar bind all five checkpoint identities, model
shapes/dtypes, source lineage, checked protocol identity, evaluation hashes,
and the audit hash. The export is not a trainer resume checkpoint.

The audit holds the supervisor output lock for its full read and requires a
terminal supervisor and drained queue/evaluator status, no worker PIDs, and no
live MECO process referencing the segment. The durable queue status anchors the
latest report in the complete evaluation inventory. This prevents checkpoint
snapshots from racing a restart or an evaluation from being silently omitted.

For a later explicit initialization path, Python callers may use
`consume_transition(path, seed=..., target_qualification_protocol=...)`. The
consumer computes current `source_identity()` itself and requires an exact
match with the source that produced the actors. It verifies the target
qualification protocol ID and file SHA against the audited transition and
checks keys, shapes, and dtypes against each current model, including when
models are supplied by the caller. It loads only the five actors, creates
new value heads and empty Adam optimizers, seeds Python and PyTorch from the
required fresh seed, issues a new run ID, and returns update/episode zero with
empty history. No old critic, optimizer, random state, run ID, or counters are
read. This helper is not wired into the supervisor or `--resume`; integration
must be reviewed separately. Existing resume identity checks are unchanged.

The audit is a consistency check over stored artifacts. It cannot prove that
the original process followed its declared protocol, recover absent history,
or establish simulator performance. The transition does not make the old
segment qualification evidence and does not by itself establish promotion.
