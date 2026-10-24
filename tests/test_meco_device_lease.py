import pytest

from frc_defense.meco_device_lease import lease_devices


def test_exclusive_contention_and_release(tmp_path):
    first = lease_devices(("cuda:0",), owner="first", lock_root=tmp_path).acquire()
    with pytest.raises(RuntimeError, match="already leased"):
        lease_devices(("cuda:0",), owner="second", lock_root=tmp_path).acquire()
    first.release()
    with lease_devices(("cuda:0",), owner="second", lock_root=tmp_path):
        pass


def test_distinct_devices_coexist_and_sort_stably(tmp_path):
    first = lease_devices(("cuda:1",), lock_root=tmp_path).acquire()
    second = lease_devices(("cuda:0",), lock_root=tmp_path).acquire()
    assert first.devices == ("cuda:1",)
    assert second.devices == ("cuda:0",)
    first.release()
    second.release()
    assert lease_devices(("cuda:1", "cuda:0"), lock_root=tmp_path).devices == (
        "cuda:0", "cuda:1")


def test_partial_acquisition_failure_releases_prior_devices(tmp_path):
    held = lease_devices(("cuda:1",), lock_root=tmp_path).acquire()
    with pytest.raises(RuntimeError, match="cuda:1"):
        lease_devices(("cuda:1", "cuda:0"), lock_root=tmp_path).acquire()
    with lease_devices(("cuda:0",), lock_root=tmp_path):
        pass
    held.release()


def test_context_releases_after_exception_and_rejects_duplicates(tmp_path):
    with pytest.raises(LookupError):
        with lease_devices(("cuda:0",), lock_root=tmp_path):
            raise LookupError("failure")
    with lease_devices(("cuda:0",), lock_root=tmp_path):
        pass
    with pytest.raises(ValueError, match="duplicate-free"):
        lease_devices(("cuda:0", "cuda:0"), lock_root=tmp_path)
