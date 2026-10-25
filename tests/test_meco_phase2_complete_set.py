import hashlib
import copy
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from frc_defense import meco_phase2_complete_set as coordinator
from frc_defense.meco_tracks import MECO_TRACKS


def _descriptor(tmp_path):
    root = tmp_path / "attempt-0001"
    outputs = coordinator._expected_outputs(root)
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    return {"schema": coordinator.SCHEMA, "descriptor_sha256": "d" * 64,
        "attempt_number": 1, "source": {}, "qualification_protocol": {
            "path": str(protocol), "id": "p", "sha256": "p" * 64},
        "scenario_id": "s", "category": "c", "seeds": [1],
        "device": "cuda:0", "outputs": outputs,
        "c_campaign_descriptor": {"campaign_id": "campaign-test", "outputs": {
            "root": str(root / coordinator.STAGE_C),
            "dataset": outputs["source_dataset"],
            "manifest": outputs["source_manifest"],
            "C1": outputs["artifacts"]["C1"],
            "C2": outputs["artifacts"]["C2"],
            "report": outputs["c_campaign_report"]}},
        "plan_a_ubp": {"output_dir": str(root / coordinator.STAGE_A)}}


def _install_adapters(monkeypatch):
    binding = {"dataset_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64, "record_count": 1}
    source_hash = "a" * 64
    c_receipts = {}
    a_receipts = {}

    def c_run(desc):
        out = desc["outputs"]
        Path(out["root"]).mkdir()
        for key in ("dataset", "C1", "C2", "report"):
            Path(out[key]).write_text(key)
        Path(out["dataset"] + ".manifest.json").write_text("manifest")

    def a_run(_dataset, _protocol, output, **_kwargs):
        output.mkdir()
        for track in ("A1", "A2", "A3"):
            path = Path(_descriptor_value["outputs"]["artifacts"][track])
            path.write_text(track)
            a_receipts[track] = {"path": str(path), "sha256": coordinator._sha_file(path)}
        bundle = Path(_descriptor_value["outputs"]["a_bundle"])
        bundle.write_text("bundle")

    def load_c(desc, *, expected_sources):
        outputs = desc["c_campaign_descriptor"]["outputs"]
        for track in ("C1", "C2"):
            path = Path(outputs[track])
            digest = coordinator._sha_file(path)
            if track in c_receipts and c_receipts[track]["artifact_sha256"] != digest:
                raise ValueError("C output changed after validation")
            c_receipts[track] = {"artifact_sha256": digest}
        return binding, dict(c_receipts)

    def load_a(desc, *, expected_sources, source_binding):
        return dict(a_receipts)

    def validate_seal(manifest, desc):
        for track in MECO_TRACKS:
            if coordinator._sha_file(Path(desc["outputs"]["artifacts"][track])) != \
                    manifest["artifacts"][track]["sha256"]:
                raise ValueError("artifact tampered")
        return manifest

    monkeypatch.setattr(coordinator, "validate_descriptor", lambda *_a, **_kw: None)
    monkeypatch.setattr(coordinator, "_load_c_stage", load_c)
    monkeypatch.setattr(coordinator, "_load_a_stage", load_a)
    monkeypatch.setattr(coordinator, "validate_complete_set_manifest", validate_seal)
    monkeypatch.setattr("frc_defense.curriculum_training.source_identity",
        lambda: {"identity": "test"})
    monkeypatch.setattr("frc_defense.meco_supervised._read_counterfactual_dataset",
        lambda *_a, **_kw: ([], source_hash, "b" * 64, None, "p", "p" * 64,
            {"seeds": [1]}))
    return c_run, a_run


_descriptor_value = None


def test_injected_restart_reuses_validated_c_stage(tmp_path, monkeypatch):
    global _descriptor_value
    descriptor = _descriptor(tmp_path)
    _descriptor_value = descriptor
    c_run, a_run = _install_adapters(monkeypatch)
    c_calls = []

    def count_c(desc):
        c_calls.append(1)
        c_run(desc)

    @contextmanager
    def lease(_devices, owner=None):
        yield

    # Simulate a process stopping after C by making A fail before creating outputs.
    def fail_a(*_args, **_kwargs):
        raise RuntimeError("injected stop after C")

    with pytest.raises(RuntimeError, match="injected stop"):
        coordinator.execute_complete_set(descriptor, c_execute=count_c,
            a_execute=fail_a, lease_factory=lease)
    first_c = next(item for item in coordinator._read_stage_journal(
        Path(descriptor["outputs"]["stage_journal"])) if item["stage"] == "C")
    result = coordinator.execute_complete_set(descriptor, c_execute=count_c,
        a_execute=a_run, lease_factory=lease)
    assert len(c_calls) == 1
    assert set(result["warm_start_artifacts"]) == set(MECO_TRACKS)
    records = coordinator._read_stage_journal(Path(descriptor["outputs"]["stage_journal"]))
    assert {record["stage"] for record in records} == {"C", "A"}
    assert next(item for item in records if item["stage"] == "C")["attempt_id"] == \
        first_c["attempt_id"]


def test_tampered_c_output_fails_closed_on_resume(tmp_path, monkeypatch):
    global _descriptor_value
    descriptor = _descriptor(tmp_path)
    _descriptor_value = descriptor
    c_run, _a_run = _install_adapters(monkeypatch)

    @contextmanager
    def lease(_devices, owner=None):
        yield

    with pytest.raises(RuntimeError):
        coordinator.execute_complete_set(descriptor, c_execute=c_run,
            a_execute=lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("stop")),
            lease_factory=lease)
    Path(descriptor["outputs"]["artifacts"]["C1"]).write_text("changed")
    with pytest.raises(ValueError, match="output changed"):
        coordinator.execute_complete_set(descriptor, c_execute=lambda _d: None,
            a_execute=lambda *_a, **_kw: None, lease_factory=lease)


def test_output_parent_symlink_created_during_lease_is_rejected(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    descriptor = _descriptor(parent)
    _install_adapters(monkeypatch)
    target = tmp_path / "other"
    target.mkdir()

    @contextmanager
    def swap_parent(_devices, owner=None):
        parent.rename(tmp_path / "original-parent")
        parent.symlink_to(target, target_is_directory=True)
        yield

    with pytest.raises(ValueError, match="symlink"):
        coordinator.execute_complete_set(descriptor, c_execute=lambda _d: None,
            a_execute=lambda *_a, **_kw: None, lease_factory=swap_parent)


def test_resume_rejects_preseeded_permissive_attempt_root(tmp_path, monkeypatch):
    descriptor = _descriptor(tmp_path)
    _install_adapters(monkeypatch)
    root = Path(descriptor["outputs"]["root"])
    root.mkdir(mode=0o777)
    root.chmod(0o777)
    Path(descriptor["outputs"]["state"]).write_text(json.dumps({
        "schema": coordinator.SCHEMA,
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "attempt_number": 1, "status": "running", "completed_stages": []}))
    leases = []

    @contextmanager
    def lease(_devices, owner=None):
        leases.append(True)
        yield

    with pytest.raises(ValueError, match="coordinator-owned mode 0700"):
        coordinator.execute_complete_set(descriptor, c_execute=lambda _d: None,
            a_execute=lambda *_a, **_kw: None, lease_factory=lease)
    assert leases == []


@pytest.mark.parametrize("hostile_file", ["state", "stage_journal", "manifest"])
def test_resume_rejects_permissive_state_or_journal(tmp_path, monkeypatch, hostile_file):
    descriptor = _descriptor(tmp_path)
    _install_adapters(monkeypatch)
    root = Path(descriptor["outputs"]["root"])
    root.mkdir(mode=0o700)
    state = Path(descriptor["outputs"]["state"])
    state.write_text(json.dumps({"schema": coordinator.SCHEMA,
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "attempt_number": 1, "status": "running", "completed_stages": []}))
    state.chmod(0o600)
    target = Path(descriptor["outputs"][hostile_file])
    if hostile_file == "stage_journal":
        target.write_text("{}\n")
    elif hostile_file == "manifest":
        target.write_text("{}")
    else:
        target.chmod(0o644)
    if hostile_file in ("stage_journal", "manifest"):
        target.chmod(0o644)
    with pytest.raises(ValueError, match="coordinator-owned private file"):
        coordinator.execute_complete_set(descriptor, c_execute=lambda _d: None,
            a_execute=lambda *_a, **_kw: None, lease_factory=lambda *_a, **_kw: None)


def test_root_replacement_during_stage_aborts_before_next_stage(tmp_path, monkeypatch):
    descriptor = _descriptor(tmp_path)
    _install_adapters(monkeypatch)

    @contextmanager
    def lease(_devices, owner=None):
        yield

    def replace_after_c(desc):
        outputs = desc["outputs"]
        output = Path(outputs["root"])
        attempt_root = output.parent
        output.mkdir()
        for key in ("dataset", "C1", "C2", "report"):
            Path(outputs[key]).write_text(key)
        Path(outputs["dataset"] + ".manifest.json").write_text("manifest")
        moved = attempt_root.with_name("detached-attempt")
        attempt_root.rename(moved)
        attempt_root.symlink_to(moved, target_is_directory=True)

    called_a = []
    with pytest.raises(ValueError, match="replaced after pinning"):
        coordinator.execute_complete_set(descriptor, c_execute=replace_after_c,
            a_execute=lambda *_a, **_kw: called_a.append(True), lease_factory=lease)
    assert called_a == []


def test_stage_journal_rejects_tampered_receipt(tmp_path):
    path = tmp_path / "stage-receipts.jsonl"
    receipt = {"schema": coordinator.SCHEMA + ".stage-receipt", "stage": "C",
        "attempt_id": "attempt-a", "status": "validated", "inputs": {}, "outputs": {}}
    coordinator._append_stage_receipt(path, receipt)
    raw = path.read_text().replace('"validated"', '"tampered"')
    path.write_text(raw)
    with pytest.raises(ValueError, match="malformed"):
        coordinator._read_stage_journal(path)


def test_seal_binding_rejects_tampered_receipt_summary(tmp_path):
    descriptor = _descriptor(tmp_path)
    outputs = descriptor["outputs"]
    source = {"dataset_sha256": "a" * 64, "dataset_manifest_sha256": "b" * 64}
    c_validators = {track: {"artifact_sha256": track.lower() * 64}
        for track in ("C1", "C2")}
    a_validators = {track: {"sha256": track.lower() * 64}
        for track in ("A1", "A2", "A3")}
    c = {"inputs": {"protocol": descriptor["qualification_protocol"],
            "seeds": descriptor["seeds"], "scenario_id": descriptor["scenario_id"],
            "category": descriptor["category"], "device": descriptor["device"]},
        "outputs": {"source_dataset": {"path": outputs["source_dataset"],
                "sha256": source["dataset_sha256"]},
            "source_manifest": {"path": outputs["source_manifest"],
                "sha256": source["dataset_manifest_sha256"]},
            "report": {"path": outputs["c_campaign_report"], "sha256": "r" * 64},
            "artifacts": {track: {"path": outputs["artifacts"][track],
                "sha256": c_validators[track]["artifact_sha256"]} for track in c_validators}},
        "validator_summary": c_validators}
    a = {"inputs": {"source_dataset": {"path": outputs["source_dataset"],
                "sha256": source["dataset_sha256"]},
            "source_manifest": {"path": outputs["source_manifest"],
                "sha256": source["dataset_manifest_sha256"]},
            "protocol": descriptor["qualification_protocol"],
            "scenario_id": descriptor["scenario_id"], "category": descriptor["category"],
            "seeds": descriptor["seeds"], "device": descriptor["device"]},
        "outputs": {"bundle": {"path": outputs["a_bundle"], "sha256": "q" * 64},
            "artifacts": {track: {"path": outputs["artifacts"][track],
                "sha256": a_validators[track]["sha256"]} for track in a_validators}},
        "validator_summary": a_validators}
    stages = {"C": {"report_sha256": "r" * 64}, "A": {"bundle_sha256": "q" * 64}}
    coordinator._validate_stage_receipt_bindings({"C": c, "A": a}, descriptor,
        stages["C"], stages["A"], source, c_validators, a_validators)
    tampered = copy.deepcopy(c)
    tampered["validator_summary"]["C1"]["artifact_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="stage journal receipts"):
        coordinator._validate_stage_receipt_bindings({"C": tampered, "A": a}, descriptor,
            stages["C"], stages["A"], source, c_validators, a_validators)


def test_manifest_warm_start_map_requires_all_five_exact_paths(tmp_path):
    from frc_defense.meco_supervisor import _manifest_warm_start_paths

    warm = {}
    expected = {}
    for track in MECO_TRACKS:
        path = tmp_path / f"{track}.pt"
        path.write_text(track)
        expected[track] = str(path.resolve())
        warm[track] = {"present": True, "binding_valid": True,
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert _manifest_warm_start_paths({"warm_starts": warm}) == expected
    del warm["A3"]
    with pytest.raises(ValueError, match="all five"):
        _manifest_warm_start_paths({"warm_starts": warm})
