# Phase 2 five-track runtime tranche

**Purpose:** produce and validate the first source-bound Plan C dataset and a
complete C1/C2/A1/A2/A3 warm-start set. This is the next runtime tranche after
the existing coordinator implementation; it is not a claim that all Phase 2
data coverage, the full MECO plan, or qualification is complete.

## Evidence at planning time

- `frc_defense/meco_phase2_complete_set.py` already provides a resumable,
  single-descriptor coordinator. It collects one matched-prefix C1/C2 dataset,
  fits both C models, calls Plan A U+B+P collection/fitting for A1/A2/A3,
  validates all five artifacts, and seals an immutable complete-set manifest.
- The current r4 campaign is still live even though its systemd unit is
  inactive. At 2026-10-25 05:33:31 UTC its queue heartbeat showed cycle 2,
  A3 running on `cuda:0`, C1 running on `cuda:1`, A1/C2 queued, A2 complete
  for the current chunk, and `stop_requested=false`. Worker PIDs 2960636 and
  3030144 were present. Do not infer that the devices are free from the
  inactive unit state; recheck live worker processes and the device-lease
  registry before any runtime action.
- r4 is diagnostic-only: scenario `pickup_simple`, category `single`,
  `evaluation_protocol_id=null`, `warm_start_artifact_sha256={}`. Its current
  checkpoint chunks do not supply supervised data or warm starts.
- The active checkout's
  `qualifications/meco/protocol.json` has protocol ID
  `dec8964454769b01fa4d79fe5f1cb7919013d94b5076e3a99a20ba5a4181f6f2` and
  SHA-256 `9314a1db838ead3e32db4ab8ed5a96f6944592fcedc162b2fe572c608afef35f`.
  Its exact `train` case for `pickup_simple/single` contains 100 seeds
  (11518889661825024 through 11518889661825420, step 4). The protocol bytes
  match the protocol used by r4. The first tranche uses these exact train
  seeds; it must never read validation/test seeds into collection or fitting.
- No Phase 2 dataset or warm-start output exists under the r4 campaign root.
  The only current evaluation is diagnostic and is not a substitute.

## Scope and result

Use one clean source worktree pinned to the reviewed current `main` at launch
time. Generate a fresh descriptor for this exact protocol, scenario, category,
seed list, indexed WX9100 device, and a new ZFS campaign attempt directory.
Run the existing complete-set coordinator once. It should produce:

- `phase2-c/matched-prefix.jsonl` and its immutable dataset manifest;
- validated C1 and C2 warm-start weights;
- the Plan A U+B dataset, independent P-route dataset, and validated A1,
  A2, and A3 weights;
- stage receipts and the final five-track complete-set manifest, with all
  readiness/qualification fields interpreted only according to their actual
  validators.

This first 100-seed `pickup_simple/single` bundle is a pipeline-enabling
matched train-split slice. It does not satisfy the plan's full scenario
coverage, prove behavior, or qualify a model. Follow-on Phase 2 coverage must
expand to the other exact train cases through additional source-bound
attempts; never pool incompatible or differently sourced bundles implicitly.

## GPU and live-run gates

1. Keep r4 uninterrupted. Do not alter its stop file, status, checkpoints,
workers, or lease records. The coordinator is not to be started until r4 has
   ended naturally or the user explicitly stops it, all worker PIDs are gone,
   and both WX9100 lease records are free.
2. Revalidate the chosen clean source commit and working tree. The r4 source
   and current output do not authorize a new-source resume. Do not attach the
   new Phase 2 evaluator or warm starts retroactively to r4.
3. Bind the coordinator to one explicit WX9100 device (`cuda:0` after fresh
   inventory); it acquires the shared device lease before output reservation
   and keeps the lease for collection and fitting. The C simulator environment
   and all model fitting must use this indexed GPU. Do not fall back to CPU
   training or CPU simulation on device errors; fail closed and preserve the
   resumable stage receipts.
4. The current Plan A U+B+P path still has CPU Python Dijkstra routing and the
   HIP exact-heap implementation is gated off. The user's GPU-only requirement
   therefore is not yet met for Plan A. Before executing this complete-set
   tranche, close the exact-heap HIP parity/profile gate and make the P-route
   collector use it without CPU fallback, or keep the coordinator blocked.
   Host-side orchestration, filesystem I/O, and serialization are expected;
   they must not silently perform model training or simulator physics on CPU.
5. Use a new numbered attempt directory; never reuse a partial directory
   unless it validates against the exact descriptor. If a CUDA or validator
   error occurs, stop that attempt at the coordinator's failure boundary,
   retain its journal, and resume only through `execute` with the same
   descriptor after the cause is fixed.

## Concrete handoff commands

Run only after gates 1–4 pass. Set `REPO` to the clean source worktree at the
reviewed launch commit and `CAMPAIGNS` to
`/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDriveCampaigns`:

```bash
set -euo pipefail
REPO='/mnt/zfs-storage/All Drives/Brian Data/Projects/.worktrees/<clean-current-source>'
CAMPAIGNS='/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDriveCampaigns'
PROTOCOL="$REPO/qualifications/meco/protocol.json"
PHASE2="$CAMPAIGNS/meco-phase2-five-track-20261025"
ATTEMPT="$PHASE2/attempt-0001"
DESCRIPTOR="$PHASE2/attempt-0001.descriptor.json"
mkdir -p "$PHASE2"
mapfile -t TRAIN_SEEDS < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); rows=[r for r in d["cases"] if r.get("scenario_id")=="pickup_simple" and r.get("category")=="single" and r.get("split")=="train"]; assert len(rows)==1 and len(rows[0]["seeds"])==100; print("\n".join(map(str,rows[0]["seeds"])))' "$PROTOCOL")
cd "$REPO"
python3 -m frc_defense.meco_phase2_complete_set describe \
  --protocol "$PROTOCOL" --scenario pickup_simple --category single \
  --seeds "${TRAIN_SEEDS[@]}" --device cuda:0 \
  --output-root "$ATTEMPT" --descriptor "$DESCRIPTOR"
python3 -m frc_defense.meco_phase2_complete_set validate "$DESCRIPTOR"
python3 -m frc_defense.meco_phase2_complete_set execute "$DESCRIPTOR"
python3 -m frc_defense.meco_phase2_complete_set warm-start-map \
  "$DESCRIPTOR" "$ATTEMPT/complete-set-manifest.json"
```

The descriptor path and attempt root are intentionally outside the source
checkout and qualification protocol. The descriptor is source-bound; if the
source commit, protocol bytes, or paths change, discard it and describe a new
attempt rather than editing it.

## Acceptance evidence

- Record the source commit/tree digest, protocol ID/file digest, descriptor
  digest, attempt path, device identity, and exact seed list.
- Verify the coordinator's stage journal proves C collection, C1/C2 fit,
  Plan A U+B/P collection and A1/A2/A3 fit; inspect each validator result and
  the complete-set manifest's exact five artifact hashes and shared bindings.
- Verify the complete set is consumable by a fresh manifest through the
  `warm-start-map` command. Start a new five-track campaign only with a newly
  validated manifest, new output roots, all five warm-start hashes, and the
  selected training/evaluation protocol bindings. It must schedule all five
  tracks across the two WX9100 devices.
- Run a seeded end-to-end simulator acceptance on the new source/checkpoints
  after launch and inspect its fresh replay/record. Before replay work, restart
  and verify the dashboard against that same checkout and confirm its served
  assets match. Report the scenario, seed, replay ID, source/checkpoint
  provenance, and dashboard currentness. A complete-set manifest alone is not
  simulation validation.
- Keep matched held-out evaluation, Phase 4, Phase 5, promotion, and
  qualification readiness false until their separate evidence gates pass.

## What this unlocks

It turns the already-reviewed coordinator from dormant code into the first
auditable real-data path: a shared C training dataset, A1/A2/A3 U+B+P
supervision, all five validated warm starts, and a fresh campaign able to
consume them. The immediate runtime gate is available WX9100 capacity; the
GPU-only Plan A route gate is a source/runtime prerequisite. Neither gate is
resolved by the active diagnostic r4 run or by this plan.
