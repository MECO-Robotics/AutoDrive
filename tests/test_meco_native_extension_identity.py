from types import SimpleNamespace

import pytest

from frc_defense import fuel_physics_hip
from frc_defense.meco_counterfactual import (
    _artifact_identity_sha256,
    _native_fuel_extension_identity,
    _verify_native_fuel_extension,
)


def test_native_extension_identity_binds_binary_and_detects_replacement(tmp_path, monkeypatch):
    binary = tmp_path / "autodrive_fuel_physics_hip.so"
    binary.write_bytes(b"native-binary-v1")
    monkeypatch.setattr(fuel_physics_hip, "_EXT", SimpleNamespace(__file__=str(binary)))

    frozen = _native_fuel_extension_identity()
    assert frozen["path"] == str(binary.resolve())
    _verify_native_fuel_extension(frozen)
    assert _artifact_identity_sha256("source-v1", frozen) != _artifact_identity_sha256(
        "source-v1", {**frozen, "sha256": "0" * 64})

    replacement = tmp_path / "replacement.so"
    replacement.write_bytes(b"native-binary-v2")
    replacement.replace(binary)
    with pytest.raises(RuntimeError, match="identity changed|path was replaced"):
        _verify_native_fuel_extension(frozen)


def test_native_extension_identity_is_none_until_loaded(monkeypatch):
    monkeypatch.setattr(fuel_physics_hip, "_EXT", None)
    assert _native_fuel_extension_identity() is None
    _verify_native_fuel_extension(None)
