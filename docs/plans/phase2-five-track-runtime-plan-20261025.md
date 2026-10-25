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
  inactive. At 2026-10-25 05:42:20 UTC its queue heartbeat showed cycle 2,
  seven completed chunks, C1 running on `cuda:1`, C2 running on `cuda:0`,
  A1 queued, A2/A3 complete for the current chunk, and
  `stop_requested=false`. Worker PIDs 3030144 and 3039903 were present. Do not
  infer that the devices are free from the inactive unit state; recheck live
  worker processes and the device-lease registry before any runtime action.
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

Use one clean source worktree pinned to the reviewed current source at launch
time. The planner worktree now includes implementation commit
`ade65b20f8aba6229d0d64207ed832ee2a1ed1ef` (“Route Plan A P replays through
gated HIP heap”); preserve it as a separate implementation commit. Generate a
fresh descriptor for this exact protocol, scenario, category, seed list,
indexed WX9100 device, and a new ZFS campaign attempt directory. Run the
existing complete-set coordinator once. It should produce:

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
3. The shared lease is keyed only by logical `cuda:N`; it does not prove which
   physical card that index names. Add and invoke an identity verifier inside
   the acquired lease and before reserving/writing the attempt directory. For
   `cuda:0`, require model `AMD Radeon Pro WX 9100`, BDF `0000:19:00.0`, PCI
   vendor/device `1002:6861`, and HIP UUID
   `30323133-6632-6261-3734-313632316134`. For `cuda:1`, require the same model,
   BDF `0000:67:00.0`, PCI vendor/device `1002:6861`, and HIP UUID
   `30323133-6661-3836-3737-346334386134`. These are the pinned identities in
   `qualifications/meco/host-gpu-inventory-20261024.json`; verify that file's
   pinned SHA before trusting it. After acquiring the shared logical lease,
   re-read runtime model/UUID with `torch.cuda.get_device_properties(index)`
   and cross-check the live BDF/vendor/device with `lspci -Dnn` (and
   `rocm-smi --showproductname --json` for the product name). Join by PCI BDF,
   not enumeration order. Fail closed if any property is unavailable or
   mismatched. Explicitly reject NVIDIA/P600 identity (`0000:b3:00.0`,
   `10de:1cb2`) even if it appears at the requested logical index. Persist the
   observed model, BDF, vendor/device, UUID, inventory digest, and logical
   lease ID with the coordinator attempt. Never treat a successful
   `cuda:0` lease alone as WX9100 authorization. The same post-lease identity
   verifier is required in `MecoSupervisor` after it acquires the two-device
   lease and before it starts any worker; otherwise the fresh campaign launch
   remains blocked even if Phase 2 collection passed its one-device check.
4. Bind the coordinator to one explicitly verified WX9100, and keep its lease
   for collection and fitting. The C simulator environment and all model
   fitting must use this verified indexed GPU. Do not fall back to CPU
   training or CPU simulation on device errors; fail closed and preserve the
   resumable stage receipts.
5. The new `ade65b20f` path can request Plan A replay/support construction
   through the gated HIP heap, but the gate must still be exercised and
   verified at runtime. Close exact-output/support-digest parity and profiling
   on the selected WX9100, then ensure the Plan A U+B/P collectors use that
   path with `require_hip_plan_a=True` and no CPU Dijkstra fallback. If any
   collector still routes on CPU, keep the complete-set coordinator blocked.
   Host orchestration, filesystem I/O, and serialization may use CPU; model
   training and simulator physics/routing may not silently do so.
6. Use a new numbered attempt directory; never reuse a partial directory
   unless it validates against the exact descriptor. If a CUDA or validator
   error occurs, stop that attempt at the coordinator's failure boundary,
   retain its journal, and resume only through `execute` with the same
   descriptor after the cause is fixed.

## Concrete handoff commands

Run only after gates 1–5 pass. Set `REPO` to the clean source worktree at the
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

## Fresh five-track campaign manifest and launch

Run this only after the complete-set manifest validates, every one of the five
weights artifacts validates against its source/protocol/track, the current
manifest-generation source is clean, all GPU-only and physical-identity gates
above pass, and r4 has ended naturally or the user explicitly stopped it. At
that point require all r4 workers to be gone and both logical shared leases to
be free. Recheck those conditions immediately before manifest validation and
again before launch. Build a fresh config from the latest reviewed settings
template, rebinding its protocol paths to the current clean checkout, its
`warm_start_artifacts` to the exact complete-set paths for A1, A2, A3, C1, and
C2, and its campaign/checkpoint/evaluation roots to a new absent namespace.
Keep the template's training split, evaluation schedule/monitor, and track
hyperparameters only after revalidating them against the current protocols;
the stale r5 config and manifest are settings inputs, not launchable artifacts.

```bash
set -euo pipefail
REPO='/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive'
CAMPAIGNS='/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDriveCampaigns'
PHASE2="$CAMPAIGNS/meco-phase2-five-track-20261025"
ATTEMPT="$PHASE2/attempt-0001"
DESCRIPTOR="$PHASE2/attempt-0001.descriptor.json"
COMPLETE_SET="$ATTEMPT/complete-set-manifest.json"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
test -f "$COMPLETE_SET"
test -f "$DESCRIPTOR"
cd "$REPO"
python3 -m frc_defense.meco_phase2_complete_set validate "$DESCRIPTOR"
TEMPLATE="$CAMPAIGNS/manifests/meco-five-continuous-20261025-r5-plan-c-audit-5389.config.json"
CONFIG="$PHASE2/five-track-r6.config.json"
R6_ROOT="$CAMPAIGNS/meco-five-continuous-20261025-r6-phase2"
# Keep the manifest outside the campaign root: the supervisor requires that
# fresh output roots remain absent until its post-lease claim.
MANIFEST="$PHASE2/five-track-r6.manifest.json"
R4_ROOT="$CAMPAIGNS/meco-five-continuous-20261025-r4"
# Run only after natural completion or an explicit user stop.
python3 - <<'PY' "$R4_ROOT/supervisor-status.json"
import json, pathlib, sys
status = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert status.get("status") in {"stopped", "complete"}, status.get("status")
assert status.get("error") is None, status.get("error")
assert all(not lane.get("pid") for lane in status.get("lanes", []))
PY
python3 - <<'PY'
import os, pathlib
needles = (b"meco-five-continuous-20261025-r4.json",
           b"AutoDriveCampaigns/meco-five-continuous-20261025-r4/checkpoints")
live = []
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) in {os.getpid(), os.getppid()}:
        continue
    try:
        args = (entry / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if any(needle in args for needle in needles):
        live.append(int(entry.name))
assert not live, f"r4 supervisor/workers are still live: {live}"
PY
cd "$REPO"
python3 - <<'PY'
from frc_defense.meco_device_lease import lease_devices
with lease_devices(["cuda:0", "cuda:1"], owner="phase2-launch-gate-probe"):
    pass
PY
python3 - <<'PY' "$TEMPLATE" "$CONFIG" "$REPO" "$ATTEMPT/complete-set-manifest.json" "$R6_ROOT"
import json, pathlib, sys
template, destination, repo, complete_path, root = map(pathlib.Path, sys.argv[1:])
c = json.loads(complete_path.read_text())
tracks = {"A1", "A2", "A3", "C1", "C2"}
assert set(c["artifacts"]) == tracks and set(c["warm_start_artifacts"]) == tracks
cfg = json.loads(template.read_text())
assert set(cfg["tracks"]) == tracks
campaign = "meco-five-continuous-20261025-r6-phase2"
assert root.name == campaign
protocol = repo / "qualifications/meco/protocol.json"
cfg["campaign_id"] = campaign
cfg["campaign_root"] = str(root)
cfg["evaluation_output"] = str(root / "evaluations")
cfg["qualification_protocol"] = str(protocol)
cfg["evaluation_protocol"] = str(protocol)
cfg["warm_start_artifacts"] = {t: c["warm_start_artifacts"][t] for t in sorted(tracks)}
for track in tracks:
    cfg["tracks"][track]["output"] = str(root / "checkpoints" / track)
destination.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
PY
# Keep manifest_output null in the manifest; capture describe's JSON stdout at
# the external path so campaign-path validation does not bind it under R6_ROOT.
python3 -m frc_defense.meco_training_manifest describe \
  --config "$CONFIG" --source-root "$REPO" > "$MANIFEST"
python3 -m frc_defense.meco_training_manifest validate \
  "$MANIFEST" --source-root "$REPO"
python3 - <<'PY' "$MANIFEST" "$ATTEMPT/complete-set-manifest.json"
import json, sys
m = json.load(open(sys.argv[1])); c = json.load(open(sys.argv[2]))
tracks = {"A1", "A2", "A3", "C1", "C2"}
assert set(m["tracks"]) == tracks
assert set(m["warm_starts"]) == tracks
assert all(m["warm_starts"][t]["present"] and
           m["warm_starts"][t]["binding_valid"] for t in tracks)
assert set(c["artifacts"]) == tracks
assert all(m["warm_starts"][t]["sha256"] == c["artifacts"][t]["sha256"] and
           m["warm_starts"][t]["path"] == c["artifacts"][t]["path"]
           for t in tracks)
assert m["readiness"]["hardware_ready"] and m["readiness"]["implementation_ready"]
print("fresh manifest binds five validated warm starts:", m["canonical_sha256"])
PY
test ! -e "$R6_ROOT"
test ! -e "$R6_ROOT/evaluations"
python3 -m frc_defense.meco_supervisor \
  --campaign-manifest "$MANIFEST" --diagnostic-launch
```

`describe` and `validate` must succeed before launch. The supervisor command is
the explicit fresh diagnostic launch path; it must not include
`--resume-campaign`. The new config must assign all five tracks to the two
verified WX9100 indices and name fresh campaign, checkpoint, and evaluation
roots. Keep `phase2_ready`, validation readiness, and qualification readiness
false unless the manifest's authoritative validators independently establish
them.

## Acceptance evidence

- Record the source commit/tree digest, protocol ID/file digest, descriptor
  digest, attempt path, device identity, and exact seed list.
- Verify the coordinator's stage journal proves C collection, C1/C2 fit,
  Plan A U+B/P collection and A1/A2/A3 fit; inspect each validator result and
  the complete-set manifest's exact five artifact hashes and shared bindings.
- Verify the complete set is consumable by a fresh manifest through the
  `warm-start-map` command and the assertions in the manifest handoff above.
  Require the manifest's pinned inventory digest to be current and the
  supervisor's post-lease runtime attestation to match both expected model,
  BDF, vendor/device, and HIP UUID tuples before accepting launch.
  In the fresh supervisor status/queue after launch, require the exact lane set
  `{A1,A2,A3,C1,C2}`, each lane bound to the intended verified WX9100, and
  each in `queued`, `running`, or `complete` state with no error/deferred
  reason. Since only two exclusive workers can run at once, do not require five
  simultaneous `running` states. Observe advancing heartbeats and completed
  training chunks until every lane has run; retain the queue/status snapshot
  proving all five were dispatched and record a completed matched frontier.
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
