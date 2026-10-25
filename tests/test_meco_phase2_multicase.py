import copy
import json
from pathlib import Path

import pytest

from frc_defense import meco_phase2_multicase as multi


@pytest.fixture
def frozen_protocol(tmp_path, monkeypatch):
    rows = [
        {"scenario_id": "case-a", "category": "single", "split": "train", "seeds": [11, 12]},
        {"scenario_id": "case-b", "category": "pair", "split": "train", "seeds": [21, 22]},
        {"scenario_id": "heldout", "category": "single", "split": "validation", "seeds": [31]},
    ]
    body = {"protocol_version": "meco-qualification-v1", "cases": rows}
    body["protocol_id"] = multi._digest(body)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(body))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(multi, "source_identity", lambda: {"runtime": "frozen-test"})
    monkeypatch.setattr(multi, "_source_snapshot", lambda _repo: ("a" * 40, "b" * 64))
    return path, repo


def _describe(tmp_path, protocol, repo, cases):
    return multi.describe_multicase(protocol_path=protocol, cases=cases,
        device="cuda:0", output_root=tmp_path / "campaign", repo_root=repo)


def test_descriptor_is_deterministic_full_case_closed_and_no_write(tmp_path, frozen_protocol):
    protocol, repo = frozen_protocol
    cases = [{"scenario_id": "case-a", "category": "single"},
        {"scenario_id": "case-b", "category": "pair"}]
    first = _describe(tmp_path, protocol, repo, cases)
    second = _describe(tmp_path, protocol, repo, cases)
    assert first == second
    assert first["coverage"] == {"status": "full_train_protocol_closure",
        "requested_case_count": 2, "protocol_train_case_count": 2,
        "full_train_protocol_closure": True}
    assert first["tracks"] == ["A1", "A2", "A3", "C1", "C2"]
    assert not (tmp_path / "campaign").exists()


@pytest.mark.parametrize("cases", [
    [{"scenario_id": "unknown", "category": "single"}],
    [{"scenario_id": "case-a", "category": "single"},
     {"scenario_id": "case-a", "category": "single"}],
    [{"scenario_id": "heldout", "category": "single"}],
    [{"scenario_id": "case-a", "category": "single", "seeds": [11, 31]}],
    [{"scenario_id": "case-a", "category": "single", "seeds": [11, 11]}],
])
def test_descriptor_rejects_invalid_case_and_seed_sets(tmp_path, frozen_protocol, cases):
    protocol, repo = frozen_protocol
    with pytest.raises(ValueError):
        _describe(tmp_path, protocol, repo, cases)


def test_descriptor_marks_selected_subset_partial_and_binds_raw_protocol(tmp_path, frozen_protocol):
    protocol, repo = frozen_protocol
    value = _describe(tmp_path, protocol, repo,
        [{"scenario_id": "case-a", "category": "single", "seeds": [11]}])
    assert value["coverage"]["status"] == "partial_train_coverage"
    assert value["ordered_cases"][0]["full_case_coverage"] is False
    assert value["qualification_protocol"]["sha256"] == multi._sha(protocol.read_bytes())
    value["candidate_k"] = 32
    body = {key: item for key, item in value.items() if key != "descriptor_sha256"}
    value["descriptor_sha256"] = multi._digest(body)
    with pytest.raises(ValueError, match="common training contract"):
        multi.validate_multicase_descriptor(value, repo_root=repo)


def test_case_coverage_rejects_missing_seed_and_duplicate_decision():
    expected = [{"scenario_id": "s", "category": "c", "seeds": [1, 2]}]
    rows = [{"scenario_id": "s", "category": "c", "seed": 1, "decision_index": 0},
        {"scenario_id": "s", "category": "c", "seed": 2, "decision_index": 0}]
    coverage = multi.case_coverage(expected, rows)
    assert coverage["s::c"]["observed_seed_count"] == 2
    with pytest.raises(ValueError, match="incomplete"):
        multi.case_coverage(expected, rows[:1])
    with pytest.raises(ValueError, match="duplicate"):
        multi.case_coverage(expected, rows + [copy.deepcopy(rows[0])])


def test_combination_is_stable_case_seed_decision_order_and_preserves_repeat_evidence(tmp_path):
    expected = [{"scenario_id": "z-case", "category": "c", "seeds": [2]},
        {"scenario_id": "a-case", "category": "c", "seeds": [1]}]
    rows = [
        {"scenario_id": "z-case", "category": "c", "seed": 2, "decision_index": 1,
            "repeats": [{"return": 3}, {"return": 4}]},
        {"scenario_id": "z-case", "category": "c", "seed": 2, "decision_index": 0,
            "repeats": [{"return": 1}, {"return": 2}]},
        {"scenario_id": "a-case", "category": "c", "seed": 1, "decision_index": 0,
            "repeats": [{"return": 5}, {"return": 5}]},
    ]
    paths = []
    for index, row_group in enumerate((rows[:2], rows[2:])):
        path = tmp_path / f"{index}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in row_group))
        paths.append(path)
    output = tmp_path / "combined.jsonl"
    result = multi.combine_jsonl_shards(paths, output, expected_cases=expected)
    actual = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(row["scenario_id"], row["decision_index"]) for row in actual] == [
        ("z-case", 0), ("z-case", 1), ("a-case", 0)]
    assert actual[0]["repeats"] == [{"return": 1}, {"return": 2}]
    assert result["record_count"] == 3


def test_combination_rejects_duplicate_cross_shard_identity_and_missing_case(tmp_path):
    one = {"scenario_id": "s", "category": "c", "seed": 1,
        "decision_index": 0, "repeat_outcomes": [1, 2]}
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first.write_text(json.dumps(one) + "\n")
    second.write_text(json.dumps(one) + "\n")
    expected = [{"scenario_id": "s", "category": "c", "seeds": [1]}]
    with pytest.raises(ValueError, match="duplicate"):
        multi.combine_jsonl_shards([first, second], tmp_path / "dup.jsonl",
            expected_cases=[expected[0], expected[0]])
    with pytest.raises(ValueError, match="one shard"):
        multi.combine_jsonl_shards([first], tmp_path / "missing.jsonl",
            expected_cases=[expected[0], {"scenario_id": "t", "category": "c", "seeds": [2]}])


def test_combination_rejects_symlink_shard(tmp_path):
    target = tmp_path / "target.jsonl"
    target.write_text(json.dumps({"scenario_id": "s", "category": "c", "seed": 1,
        "decision_index": 0}) + "\n")
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        multi.combine_jsonl_shards([link], tmp_path / "out.jsonl",
            expected_cases=[{"scenario_id": "s", "category": "c", "seeds": [1]}])


def test_case_receipt_recomputes_cross_target_bindings(tmp_path):
    runtime = {"runtime": "source"}
    device = {"device": "cuda:0", "identity": "wx9100-test"}
    descriptor = {"descriptor_sha256": "d" * 64,
        "source": {"commit": "c" * 40, "source_tree_sha256": "t" * 64},
        "runtime_source_identity": runtime,
        "qualification_protocol": {"path": str(tmp_path / "protocol.json"),
            "id": "p" * 64, "sha256": "q" * 64}, "device": "cuda:0",
        "prediction_mode": "cv"}
    shard = {"case_key": ["s", "c"], "seeds": [1]}
    files = {}
    trajectory_manifest = None
    for family in ("trajectory", "ub", "p"):
        row = {"scenario_id": "s", "category": "c", "seed": 1,
            "decision_index": 0, "protocol_id": "p" * 64,
            "qualification_protocol_sha256": "q" * 64,
            "source_identity_sha256": multi._digest(runtime), "device": device}
        data_path = tmp_path / f"{family}.jsonl"
        data_path.write_text(json.dumps(row) + "\n")
        manifest = {"split": "train", "seeds": [1], "protocol_id": "p" * 64,
            "qualification_protocol_sha256": "q" * 64, "sources": runtime,
            "source_identity_sha256": multi._digest(runtime), "device": device,
            "prediction_mode": "cv", "record_count": 1,
            "dataset_sha256": multi._sha(data_path.read_bytes())}
        if family != "ub":
            manifest.update({"scenario_id": "s", "category": "c"})
        if family == "ub":
            manifest.pop("seeds")
        manifest_path = tmp_path / f"{family}.manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        files[family] = {"path": str(data_path), "sha256": multi._sha(data_path.read_bytes()),
            "manifest_path": str(manifest_path),
            "manifest_sha256": multi._sha(manifest_path.read_bytes())}
        if family == "trajectory":
            trajectory_manifest = (data_path, manifest_path)
    source_path, source_manifest_path = trajectory_manifest
    for family in ("ub", "p"):
        manifest_path = Path(files[family]["manifest_path"])
        manifest = json.loads(manifest_path.read_text())
        manifest.update({"source_dataset_sha256": multi._sha(source_path.read_bytes()),
            "source_manifest_sha256": multi._sha(source_manifest_path.read_bytes())})
        manifest_path.write_text(json.dumps(manifest))
        files[family]["manifest_sha256"] = multi._sha(manifest_path.read_bytes())
    receipt = {"schema": multi.SCHEMA + ".case-receipt",
        "descriptor_sha256": descriptor["descriptor_sha256"], "case_key": ["s", "c"],
        "seeds": [1], "source": descriptor["source"],
        "qualification_protocol": descriptor["qualification_protocol"],
        "device": "cuda:0", "datasets": files}
    assert multi.validate_case_receipt(receipt, shard, descriptor) == receipt
    p_manifest_path = Path(files["p"]["manifest_path"])
    p_manifest = json.loads(p_manifest_path.read_text())
    p_manifest["source_dataset_sha256"] = "x" * 64
    p_manifest_path.write_text(json.dumps(p_manifest))
    files["p"]["manifest_sha256"] = multi._sha(p_manifest_path.read_bytes())
    with pytest.raises(ValueError, match="trajectory shard"):
        multi.validate_case_receipt(receipt, shard, descriptor)
    with pytest.raises(ValueError, match="complete-set validator evidence"):
        multi.validate_case_receipt(receipt, shard, descriptor, require_complete_set=True)


def test_combined_plan_a_manifest_retains_each_source_triplet(tmp_path):
    descriptor = {"descriptor_sha256": "d" * 64, "case_set_sha256": "k" * 64,
        "source": {"commit": "c" * 40, "source_tree_sha256": "t" * 64},
        "qualification_protocol": {"path": "protocol.json", "id": "p" * 64,
            "sha256": "q" * 64}, "runtime_source_identity": {"r": "s"},
        "device": "cuda:0", "prediction_mode": "cv"}
    cases = [{"scenario_id": "s1", "category": "single", "seeds": [1]},
        {"scenario_id": "s2", "category": "pair", "seeds": [2]}]
    source_shards = []
    target_shards = []
    for index, case in enumerate(cases):
        source = tmp_path / f"source-{index}.jsonl"
        source.write_text(json.dumps({**case, "seed": case["seeds"][0],
            "decision_index": 0}) + "\n")
        source_manifest = {"schema": "source-v1", "target_schema": "source-target-v1",
            "split": "train", **case, "protocol_id": "p" * 64,
            "qualification_protocol_sha256": "q" * 64,
            "sources": descriptor["runtime_source_identity"],
            "source_identity_sha256": multi._digest(descriptor["runtime_source_identity"]),
            "device": {"device": "cuda:0"}, "prediction_mode": "cv",
            "prediction_contract": {"mode": "cv"}, "record_count": 1,
            "dataset_sha256": multi._sha(source.read_bytes())}
        source_sidecar = source.with_suffix(".jsonl.manifest.json")
        source_sidecar.write_text(json.dumps(source_manifest))
        source_shards.append({"path": str(source), "manifest_path": str(source_sidecar)})
        target = tmp_path / f"ub-{index}.jsonl"
        target.write_text(json.dumps({**case, "seed": case["seeds"][0],
            "decision_index": 0}) + "\n")
        target_manifest = {"schema": "ub-v1", "target_schema": "ub-target-v1",
            "split": "train", **case, "protocol_id": "p" * 64,
            "qualification_protocol_sha256": "q" * 64,
            "sources": descriptor["runtime_source_identity"],
            "source_identity_sha256": multi._digest(descriptor["runtime_source_identity"]),
            "device": {"device": "cuda:0"}, "prediction_mode": "cv",
            "prediction_contract": {"mode": "cv"}, "record_count": 1,
            "dataset_sha256": multi._sha(target.read_bytes()),
            "source_dataset_sha256": multi._sha(source.read_bytes()),
            "source_manifest_sha256": multi._sha(source_sidecar.read_bytes())}
        target_sidecar = target.with_suffix(".jsonl.manifest.json")
        target_sidecar.write_text(json.dumps(target_manifest))
        target_shards.append(target)
    trajectory = tmp_path / "combined-trajectory.jsonl"
    trajectory.write_text("pooled trajectory\n")
    trajectory_manifest = trajectory.with_suffix(".jsonl.manifest.json")
    trajectory_manifest.write_text("pooled manifest\n")
    result = multi.combine_target_shards("ub", target_shards, tmp_path / "combined-ub.jsonl",
        expected_cases=cases, descriptor=descriptor,
        source_binding={"path": str(trajectory), "sha256": multi._sha(trajectory.read_bytes()),
            "manifest_sha256": multi._sha(trajectory_manifest.read_bytes())},
        source_shards=source_shards)
    pooled = json.loads(Path(result["manifest_path"]).read_text())
    assert [row["source_dataset_sha256"] for row in pooled["case_source_bindings"]] == [
        multi._sha(Path(row["path"]).read_bytes()) for row in source_shards]
    assert [row["source_manifest_sha256"] for row in pooled["case_source_bindings"]] == [
        multi._sha(Path(row["manifest_path"]).read_bytes()) for row in source_shards]
    assert pooled["source_dataset_sha256"] == multi._sha(trajectory.read_bytes())
