# MECO held-out validation frontier

`frc_defense.meco_heldout_frontier` seals the inputs for one validation-only
campaign. It is a provisional input lock, not a result, qualification, or
permission to consume final-test seeds. The compiler requires one checkpoint and
configuration hash for each of A1, A2, A3, C1, and C2, plus a source identity.
For each checkpoint it reads the supported MECO `schema_version: 1` payload,
requires its embedded `config.track` to match the assigned track, and recomputes
the immutable configuration digest from that config with only `updates`
excluded, using canonical JSON with sorted keys and `allow_nan=False`. It also
requires all five payloads to bind one shared nonempty training protocol ID. A
supplied digest that does not match is rejected. Legacy curriculum checkpoints
and supervisor bundles are not supported.

It binds those values to the exact protocol bytes and the deterministic
validation campaign's per-job seed-manifest and policy/category/seed coverage
digests. `qualification_claim` and `qualification_eligible` remain false.

The contrast roster covers the plan's A1/A2/A3/C1/C2 comparisons, D0/D1/D2
controls, matched current versus predictive opponent information, A-track P=0,
Plan C K=8/16/32, and inference-disable fallback. D3 remains explicitly
unsupported because it is optional in the plan and absent from the registered
schema-7 matrix.

Create the single root JSON only after all five final evaluation checkpoints,
their configuration hashes, and a frozen source-identity JSON are ready:

```sh
python3 scripts/meco_heldout_frontier.py compile \
  --protocol /path/to/protocol.json \
  --source-identity-json /path/to/source-identity.json \
  --training-configuration-hashes-json /path/to/config-hashes.json \
  --a1-checkpoint /path/to/A1.pt --a2-checkpoint /path/to/A2.pt \
  --a3-checkpoint /path/to/A3.pt --c1-checkpoint /path/to/C1.pt \
  --c2-checkpoint /path/to/C2.pt \
  --frontier /path/to/output/heldout-frontier.json
```

The compiler creates the root file exclusively and refuses to replace it. Verify
it later with the same command arguments and `verify` in place of `compile`.
Verification rebuilds the campaign and rejects any protocol, source,
checkpoint, configuration, manifest, or coverage drift. This tooling does not
run a simulator, launch training, or generate a frontier from partial live
checkpoints. Deserialization uses PyTorch's `weights_only=False` loader, as in
the repository's MECO checkpoint readers; use trusted local training artifacts.
