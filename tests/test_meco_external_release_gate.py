from __future__ import annotations

import os
from pathlib import Path

import pytest

from frc_defense import meco_external_release_gate as release_gate


def _campaign():
    return {"worker_assignments": [
        {"worker_id": "wx0", "device": "cuda:0",
         "device_uuid": release_gate.WX9100_BY_INDEX["cuda:0"]["uuid"],
         "device_model": "AMD Radeon Pro WX 9100", "backend": "hip"},
        {"worker_id": "wx1", "device": "cuda:1",
         "device_uuid": release_gate.WX9100_BY_INDEX["cuda:1"]["uuid"],
         "device_model": "AMD Radeon Pro WX 9100", "backend": "hip"},
    ]}


def _proc(proc_root: Path, pid: int, argv: tuple[str, ...]):
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"\0".join(x.encode() for x in argv) + b"\0")
    (process / "fd").mkdir()
    return process


def _empty_system(tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    sys_root = tmp_path / "sys"
    (sys_root / "class/drm").mkdir(parents=True)
    return proc_root, sys_root


def test_release_observation_requires_both_assigned_wx9100_uuids():
    campaign = _campaign()
    campaign["worker_assignments"][0]["device_uuid"] = "00000000-0000-0000-0000-000000000000"
    with pytest.raises(ValueError, match="indexed WX9100 UUID"):
        release_gate.verify_external_release_before_paired_run(campaign)


def test_release_observation_rejects_canonical_r4_supervisor_and_spawned_trainer(tmp_path):
    proc_root, sys_root = _empty_system(tmp_path)
    cases = [
        (101, ("python", "-m", "frc_defense.meco_supervisor",
            "--campaign-manifest", str(release_gate.R4_MANIFEST_PATH))),
        (102, ("python", "-m", "frc_defense.meco_track_training", "train",
            "--output", str(release_gate.R4_CAMPAIGN_ROOT / "checkpoints/A1"))),
    ]
    for pid, argv in cases:
        _proc(proc_root, pid, argv)
        with pytest.raises(RuntimeError, match="canonical r4"):
            release_gate.verify_external_release_before_paired_run(
                _campaign(), proc_root=proc_root, sys_root=sys_root,
                settle_seconds=0)
        (proc_root / str(pid)).rename(proc_root / f"stopped-{pid}")


def test_release_observation_rejects_open_handle_to_indexed_wx9100(tmp_path):
    proc_root, sys_root = _empty_system(tmp_path)
    process = _proc(proc_root, 201, ("python", "unrelated-task"))
    (process / "fd/7").symlink_to("/dev/dri/renderD128")
    device_link = sys_root / "class/drm/renderD128/device"
    device_link.parent.mkdir(parents=True, exist_ok=True)
    target = sys_root / "devices/pci0000:00/0000:19:00.0"
    target.mkdir(parents=True)
    device_link.symlink_to(target)
    with pytest.raises(RuntimeError, match="active DRM process ownership"):
        release_gate.verify_external_release_before_paired_run(
            _campaign(), proc_root=proc_root, sys_root=sys_root,
            settle_seconds=0)


def test_release_observation_rejects_unreadable_process_ownership(tmp_path, monkeypatch):
    proc_root, sys_root = _empty_system(tmp_path)
    process = _proc(proc_root, 202, ("python", "unrelated-task"))
    fd_path = process / "fd/4"
    fd_path.touch()
    real_readlink = os.readlink

    def guarded_readlink(path, *args, **kwargs):
        if Path(path) == fd_path:
            raise PermissionError("hidden file descriptor")
        return real_readlink(path, *args, **kwargs)

    monkeypatch.setattr(release_gate.os, "readlink", guarded_readlink)
    with pytest.raises(RuntimeError, match="device ownership"):
        release_gate.verify_external_release_before_paired_run(
            _campaign(), proc_root=proc_root, sys_root=sys_root,
            settle_seconds=0)


def test_release_observation_emits_two_empty_read_only_observations(tmp_path):
    proc_root, sys_root = _empty_system(tmp_path)
    sleeps = []
    now = [10.0]

    def fake_sleep(duration):
        sleeps.append(duration)
        now[0] += duration

    observation = release_gate.verify_external_release_before_paired_run(
        _campaign(), proc_root=proc_root, sys_root=sys_root,
        settle_seconds=0.25, clock=lambda: now[0], sleep=fake_sleep)
    receipt = observation.record()
    assert sleeps == [0.25]
    assert receipt["schema"] == "meco-external-release-observation-v1"
    assert receipt["campaign_id"] == release_gate.R4_CAMPAIGN_ID
    assert len(receipt["observations"]) == 2
    assert all(not row["r4_processes"] for row in receipt["observations"])
    assert all(not any(row["device_owners"].values())
               for row in receipt["observations"])
    assert receipt["devices"]["cuda:0"]["uuid"] == release_gate.WX9100_BY_INDEX[
        "cuda:0"]["uuid"]
    assert release_gate.validate_persisted_release_observation_record(
        receipt, _campaign()) == receipt["observation_sha256"]
    with pytest.raises(TypeError, match="only come from the observer"):
        release_gate._LiveReleaseCapability(object(), receipt)
