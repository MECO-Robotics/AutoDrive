from types import SimpleNamespace

import pytest

from frc_defense import meco_deployment
from frc_defense.meco_tracks import MECO_TRACKS


def _capture_names(directory):
    for track in MECO_TRACKS:
        (directory / f"{track}-generation-00000001.npz").write_bytes(b"fixture")


def test_capture_profile_requires_exactly_one_capture_for_each_track(tmp_path):
    _capture_names(tmp_path)

    found = meco_deployment._capture_set(tmp_path)

    assert tuple(found) == MECO_TRACKS
    assert all(path.is_file() for path in found.values())


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected"])
def test_capture_profile_rejects_incomplete_duplicate_or_unexpected_sets(
        tmp_path, mutation):
    _capture_names(tmp_path)
    if mutation == "missing":
        (tmp_path / "C2-generation-00000001.npz").unlink()
    elif mutation == "duplicate":
        (tmp_path / "A1-generation-00000002.npz").write_bytes(b"fixture")
    else:
        (tmp_path / "OTHER-generation-00000001.npz").write_bytes(b"fixture")

    with pytest.raises((ValueError, FileNotFoundError)):
        meco_deployment._capture_set(tmp_path)


def test_capture_profile_rejects_any_device_pair_other_than_both_wx9100s(
        tmp_path, monkeypatch):
    def unexpected_lease(*_args, **_kwargs):
        pytest.fail("invalid device pair reached lease acquisition")

    monkeypatch.setattr("frc_defense.meco_device_lease.lease_devices", unexpected_lease)

    with pytest.raises(ValueError, match="exactly --device cuda:0 --device cuda:1"):
        meco_deployment.production_capture_profile_report(
            tmp_path, tmp_path, ("cuda:0",), iterations=1, warmup=0)


def test_profiler_source_identity_rejects_dirty_checkout(monkeypatch):
    responses = iter(("a" * 40 + "\n", " M modified.py\n"))

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(stdout=next(responses))

    monkeypatch.setattr(meco_deployment.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="clean profiler checkout"):
        meco_deployment._profiler_source_identity()
