import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import shutil

import pytest

from frc_defense.meco_plan_a_ubp import orchestrate_plan_a_ubp
from frc_defense.meco_plan_a_ubp import _write_bundle_reports


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, *, defect=None, lease_events=None, swap_parent=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"protocol":"frozen"}\n')
    source = tmp_path / "source.jsonl"
    source.write_text('{"source":"trajectory"}\n')
    protocol_sha = _sha(protocol)
    sources = {"runtime": "fixture-source"}
    source_manifest = {
        "dataset_sha256": _sha(source), "qualification_protocol_sha256": protocol_sha,
        "split": "train", "seeds": [11, 12], "scenario_id": "fixture",
        "category": "baseline", "protocol_id": "fixture-protocol",
        "sources": sources, "source_identity_sha256": "identity-hash",
        "device": {"device": "cuda:0", "name": "fixture-gpu"},
        "prediction_mode": "direct", "prediction_contract": {"mode": "direct"},
    }
    _write_json(source.with_suffix(source.suffix + ".manifest.json"), source_manifest)
    source_manifest_sha = _sha(source.with_suffix(source.suffix + ".manifest.json"))
    shared = {
        "split": "train", "protocol_id": "fixture-protocol",
        "qualification_protocol_sha256": protocol_sha, "sources": sources,
        "source_identity_sha256": "identity-hash", "device": source_manifest["device"],
        "prediction_mode": "direct", "prediction_contract": {"mode": "direct"},
    }

    def collect_ub(output, *, dataset, qualification_protocol, device):
        rows = []
        for seed in [11, 12]:
            rows.append({"scenario_id": "fixture", "category": "baseline", "seed": seed,
                "selected_node_coverage_fraction": 0.25,
                "node_u_targets": [{"status": "available", "u_target": 1.0}],
                "behavior_preferences": [{"status": "comparable"}]})
        if defect == "missing_u":
            rows[0]["node_u_targets"] = []
        if defect == "missing_b":
            rows[0]["behavior_preferences"] = []
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest = {**shared, "source_dataset_sha256": _sha(source),
            "source_manifest_sha256": source_manifest_sha, "dataset_sha256": _sha(output),
            "expert_goal_source": None}
        if defect == "mismatch_protocol":
            manifest["qualification_protocol_sha256"] = "bad"
        _write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
        if defect == "tamper_source":
            source.write_text(source.read_text() + "changed\n")

    def collect_p(output, *, dataset, qualification_protocol, p_cost_scale,
            intervention_maps, device):
        rows = []
        for seed in [11, 12]:
            outcomes = [
                {"native_route": {"route_trace_sha256": f"route-{seed}-a",
                    "independent_cost_oracle": {"route_cost_grid_edges": 1.0}}},
                {"native_route": {"route_trace_sha256": f"route-{seed}-b",
                    "independent_cost_oracle": {"route_cost_grid_edges": 2.0}}},
            ]
            if defect == "missing_p":
                outcomes = outcomes[:1]
            rows.append({"seed": seed, "plan_a_p_route_groups": [{"outcomes": outcomes}]})
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest = {**shared, "seeds": [11, 12], "scenario_id": "fixture",
            "category": "baseline", "source_counterfactual_dataset_sha256": _sha(source),
            "source_counterfactual_manifest_sha256": source_manifest_sha,
            "p_intervention_scale": p_cost_scale, "dataset_sha256": _sha(output),
            "record_count": 2}
        _write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)

    def fit(track, legal_dataset, source_dataset, p_dataset, output, **kwargs):
        output.write_bytes(f"checkpoint:{track}".encode())
        if defect == "tamper_dataset" and track == "A1":
            legal_dataset.write_text(legal_dataset.read_text() + "{}\n")
        if defect == "bad_artifact" and track == "A2":
            output.write_bytes(b"")

    def validate(track, checkpoint, **kwargs):
        if defect == "bad_artifact" and track == "A2":
            raise ValueError("invalid mocked checkpoint")
        assert checkpoint.is_file() and checkpoint.stat().st_size > 0

    def rebind(checkpoint, **kwargs):
        if defect == "rebind_failure":
            raise ValueError("mock rebind failure")
        # Mock checkpoints contain no embedded dataset paths.
        assert checkpoint.is_file()

    def prepare(track, checkpoint, *, staged_legal, staged_source, staged_p,
            final_legal, final_source, final_p, rebind, validate, **kwargs):
        validation_copy = checkpoint.with_name(f".{checkpoint.name}.validation")
        try:
            shutil.copyfile(checkpoint, validation_copy)
            rebind(validation_copy, legal_dataset=staged_legal,
                source_dataset=staged_source, p_dataset=staged_p)
            validate(track, validation_copy)
        finally:
            validation_copy.unlink(missing_ok=True)
        rebind(checkpoint, legal_dataset=final_legal,
            source_dataset=final_source, p_dataset=final_p)

    def write_reports(staging, bundle, report):
        if defect == "write_failure":
            raise OSError("mock report write failure")
        _write_bundle_reports(staging, bundle, report)

    @contextmanager
    def lease_devices(devices, *, owner):
        if lease_events is not None:
            lease_events.append(("acquire", tuple(devices), owner))
        if defect == "lease_failure":
            raise RuntimeError("mock lease busy")
        try:
            if defect == "swap_parent":
                parent, moved, replacement = swap_parent
                parent.rename(moved)
                replacement.mkdir()
                parent.symlink_to(replacement, target_is_directory=True)
            yield
        finally:
            if lease_events is not None:
                lease_events.append(("release", tuple(devices), owner))

    return source, protocol, {"collect_ub": collect_ub, "collect_p": collect_p,
        "fit": fit, "validate": validate, "rebind": rebind,
        "prepare": prepare, "write_reports": write_reports,
        "lease_devices": lease_devices}


def _run(tmp_path, defect=None, *, output_name="bundle"):
    source, protocol, adapters = _fixture(tmp_path, defect=defect)
    return orchestrate_plan_a_ubp(source, protocol, tmp_path / output_name,
        device="cuda:0", _adapters=adapters)


def test_publishes_validated_three_track_bundle_with_bound_hashes_and_sampled_coverage(tmp_path):
    result = _run(tmp_path)
    bundle_path = Path(result["path"])
    bundle = json.loads(bundle_path.read_text())
    assert result["artifact_count"] == 3
    assert bundle["binding"]["seeds"] == [11, 12]
    assert bundle["collected_datasets"][0]["target_counts"]["u_targets"] == 2
    assert bundle["collected_datasets"][0]["target_counts"]["b_preferences"] == 2
    assert bundle["collected_datasets"][1]["target_counts"]["non_tied_route_pairs"] == 2
    assert bundle["collected_datasets"][0]["sampled_selected_node_coverage_fraction"] == 0.25
    assert bundle["expert_u_claim"] is False
    assert bundle["exhaustive_legal_node_coverage_claim"] is False
    assert bundle["phase2_ready"] is False
    assert bundle["phase2_complete"] is False
    assert [item["track"] for item in bundle["artifacts"]] == ["A1", "A2", "A3"]
    report = json.loads((bundle_path.parent / "report.json").read_text())
    assert report["phase2_ready"] is False
    assert report["phase2_complete"] is False
    assert result["phase2_ready"] is False
    assert result["phase2_complete"] is False


@pytest.mark.parametrize("defect, message", [
    ("mismatch_protocol", "differs from immutable source binding"),
    ("missing_u", "each frozen train seed"),
    ("missing_b", "each frozen train seed"),
    ("missing_p", "too few non-tied oracle route pairs"),
    ("bad_artifact", "invalid mocked checkpoint"),
    ("rebind_failure", "mock rebind failure"),
    ("write_failure", "mock report write failure"),
    ("tamper_source", "source dataset changed during U\\+B collection"),
    ("tamper_dataset", "changed during fitting"),
])
def test_failure_never_publishes_partial_bundle(tmp_path, defect, message):
    with pytest.raises((ValueError, RuntimeError, OSError), match=message):
        _run(tmp_path, defect)
    assert not (tmp_path / "bundle").exists()
    assert not (tmp_path / ".bundle.claim").exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_bundle_bindings_are_deterministic_for_identical_inputs(tmp_path):
    first = _run(tmp_path / "first", output_name="bundle")
    second = _run(tmp_path / "second", output_name="bundle")
    first_bundle = json.loads(Path(first["path"]).read_text())
    second_bundle = json.loads(Path(second["path"]).read_text())
    assert first_bundle["binding"] == second_bundle["binding"]
    assert {key: value for key, value in first_bundle["inputs"].items()
        if key.endswith("sha256")} == {key: value for key, value in second_bundle["inputs"].items()
        if key.endswith("sha256")}
    assert first_bundle["artifacts"] == second_bundle["artifacts"]


def test_refuses_to_overwrite_published_bundle(tmp_path):
    _run(tmp_path)
    with pytest.raises(FileExistsError):
        _run(tmp_path)


def test_lease_is_acquired_before_output_claim_and_covers_full_workflow(tmp_path):
    events = []
    source, protocol, adapters = _fixture(tmp_path, lease_events=events)
    result = orchestrate_plan_a_ubp(source, protocol, tmp_path / "bundle",
        device="cuda:0", _adapters=adapters)
    assert result["status"] == "complete_validated_artifact_bundle"
    assert events[0] == ("acquire", ("cuda:0",), "meco-plan-a-ubp-orchestrator")
    assert events[-1] == ("release", ("cuda:0",), "meco-plan-a-ubp-orchestrator")


def test_lease_acquisition_failure_creates_no_output_claim_or_staging(tmp_path):
    source, protocol, adapters = _fixture(tmp_path, defect="lease_failure")
    output = tmp_path / "bundle"
    with pytest.raises(RuntimeError, match="mock lease busy"):
        orchestrate_plan_a_ubp(source, protocol, output, device="cuda:0",
            _adapters=adapters)
    assert not output.exists()
    assert not (tmp_path / ".bundle.claim").exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_parent_symlink_swap_during_lease_wait_cannot_redirect_bundle_writes(tmp_path):
    source, protocol, _ = _fixture(tmp_path / "inputs")
    parent = tmp_path / "output-parent"
    parent.mkdir()
    moved = tmp_path / "moved-parent"
    replacement = tmp_path / "replacement-parent"
    _, _, adapters = _fixture(tmp_path / "mock-inputs", defect="swap_parent",
        swap_parent=(parent, moved, replacement))
    output = parent / "bundle"
    with pytest.raises(RuntimeError, match="parent changed while waiting"):
        orchestrate_plan_a_ubp(source, protocol, output, device="cuda:0",
            _adapters=adapters)
    assert parent.is_symlink()
    assert not (replacement / "bundle").exists()
    assert not list(replacement.iterdir())
    assert not list(moved.glob(".bundle.claim"))
    assert not list(moved.glob(".bundle.staging-*"))


def test_parent_symlink_swap_during_publication_removes_owned_bundle(tmp_path, monkeypatch):
    from frc_defense import meco_plan_a_ubp

    source, protocol, adapters = _fixture(tmp_path / "inputs")
    parent = tmp_path / "output-parent"
    parent.mkdir()
    moved = tmp_path / "moved-parent"
    replacement = tmp_path / "replacement-parent"
    replacement.mkdir()
    output = parent / "bundle"
    original_rename = meco_plan_a_ubp._rename_noreplace_at

    def rename_then_swap(parent_fd, source_name, destination_name):
        original_rename(parent_fd, source_name, destination_name)
        parent.rename(moved)
        parent.symlink_to(replacement, target_is_directory=True)

    monkeypatch.setattr(meco_plan_a_ubp, "_rename_noreplace_at", rename_then_swap)
    with pytest.raises(RuntimeError, match="parent changed during publication"):
        orchestrate_plan_a_ubp(source, protocol, output, device="cuda:0",
            _adapters=adapters)

    assert parent.is_symlink()
    assert not (replacement / "bundle").exists()
    assert not list(replacement.iterdir())
    assert not (moved / "bundle").exists()
    assert not (moved / ".bundle.claim").exists()
    assert not list(moved.glob(".bundle.staging-*"))


@pytest.mark.parametrize("which", ["source", "protocol"])
def test_rejects_output_overlapping_source_or_protocol_before_lease(tmp_path, which):
    source, protocol, adapters = _fixture(tmp_path)
    with pytest.raises(ValueError, match="overlaps protected path"):
        orchestrate_plan_a_ubp(source, protocol, source if which == "source" else protocol,
            device="cuda:0", _adapters=adapters)


def test_rejects_output_overlapping_pinned_live_root(tmp_path):
    from frc_defense.meco_training_manifest import PINNED_LIVE_ROOTS
    source, protocol, adapters = _fixture(tmp_path)
    output = Path(PINNED_LIVE_ROOTS[0]) / "plan-a-orchestrator-test"
    with pytest.raises(ValueError, match="overlaps protected path"):
        orchestrate_plan_a_ubp(source, protocol, output, device="cuda:0",
            _adapters=adapters)


def test_rejects_output_inside_repository(tmp_path):
    from frc_defense.meco_training_manifest import ROOT
    source, protocol, adapters = _fixture(tmp_path)
    output = Path(ROOT) / "generated-plan-a-test-output"
    with pytest.raises(ValueError, match="overlaps protected path"):
        orchestrate_plan_a_ubp(source, protocol, output, device="cuda:0",
            _adapters=adapters)


def test_rejects_symlink_ancestor_before_lease(tmp_path):
    source, protocol, adapters = _fixture(tmp_path)
    target = tmp_path / "real-parent"
    target.mkdir()
    link = tmp_path / "linked-parent"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink ancestor"):
        orchestrate_plan_a_ubp(source, protocol, link / "bundle", device="cuda:0",
            _adapters=adapters)
