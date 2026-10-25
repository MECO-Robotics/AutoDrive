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


def _execute_adapters(tmp_path, descriptor, *, collector_hook=None, combine_hook=None,
        event_log=None):
    """Synthetic strict receipts/sidecars for execute_multicase's restart contract."""
    runtime = descriptor["runtime_source_identity"]
    protocol = descriptor["qualification_protocol"]
    collected = []
    events = event_log if event_log is not None else []

    def collect(case, attempt_root):
        case_key = case["case_key"]
        events.append(("collect", tuple(case_key), str(attempt_root)))
        attempt_root.mkdir(parents=True, mode=0o700)
        if collector_hook:
            collector_hook(case, attempt_root)
        collected.append((case_key, str(attempt_root)))
        manifest_path = attempt_root / "complete.json"
        complete_manifest = {"synthetic_complete": True}
        manifest_path.write_text(json.dumps(complete_manifest))
        one_case = {"scenario_id": case_key[0], "category": case_key[1],
            "seeds": case["seeds"], "device": descriptor["device"],
            "source": descriptor["source"], "qualification_protocol": protocol,
            "outputs": {"root": str(attempt_root), "manifest": str(manifest_path)}}
        datasets = {}
        source = None
        for family in ("trajectory", "ub", "p"):
            data_path = attempt_root / f"{family}.jsonl"
            device_info = {"device": descriptor["device"], "synthetic": True}
            rows = [{"scenario_id": case_key[0], "category": case_key[1],
                "seed": seed, "decision_index": 0,
                "protocol_id": protocol["id"],
                "qualification_protocol_sha256": protocol["sha256"],
                "source_identity_sha256": multi._digest(runtime),
                "device": device_info} for seed in case["seeds"]]
            data_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            manifest = {"split": "train", "scenario_id": case_key[0],
                "category": case_key[1], "seeds": case["seeds"],
                "protocol_id": protocol["id"],
                "qualification_protocol_sha256": protocol["sha256"],
                "sources": runtime, "source_identity_sha256": multi._digest(runtime),
                "device": device_info, "prediction_mode": descriptor["prediction_mode"],
                "record_count": len(rows), "dataset_sha256": multi._sha(data_path.read_bytes())}
            if family != "trajectory":
                manifest.update({"source_dataset_sha256": source["sha256"],
                    "source_manifest_sha256": source["manifest_sha256"]})
            sidecar = data_path.with_suffix(data_path.suffix + ".manifest.json")
            sidecar.write_text(json.dumps(manifest, sort_keys=True))
            datasets[family] = {"path": str(data_path), "sha256": multi._sha(data_path.read_bytes()),
                "manifest_path": str(sidecar), "manifest_sha256": multi._sha(sidecar.read_bytes())}
            if family == "trajectory":
                source = datasets[family]
        return {"schema": multi.SCHEMA + ".case-receipt",
            "descriptor_sha256": descriptor["descriptor_sha256"],
            "case_key": case_key, "seeds": case["seeds"],
            "source": descriptor["source"], "qualification_protocol": protocol,
            "device": descriptor["device"], "one_case_descriptor": one_case,
            "attempt_root": str(attempt_root),
            "complete_set_manifest_path": str(manifest_path),
            "complete_set_manifest_sha256": multi._sha(manifest_path.read_bytes()),
            "complete_set_manifest": complete_manifest, "datasets": datasets}

    def combine(family, shards, output, *, expected_cases, descriptor,
            source_binding=None, source_shards=None):
        records = []
        shard_manifests = []
        for case, shard_path in zip(expected_cases, shards):
            path = Path(shard_path)
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            records.extend(rows)
            sidecar = path.with_suffix(path.suffix + ".manifest.json")
            shard_manifests.append({"case_key": [case["scenario_id"], case["category"]],
                "seeds": case["seeds"], "path": str(path), "manifest_path": str(sidecar),
                "dataset_sha256": multi._sha(path.read_bytes()),
                "manifest_sha256": multi._sha(sidecar.read_bytes()),
                "record_count": len(rows), "schema": "synthetic-v1", "target_schema": "synthetic-v1"})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
        coverage = multi.case_coverage(expected_cases, records)
        manifest = {"schema": multi.SCHEMA + ".combined-target", "target_family": family,
            "descriptor_sha256": descriptor["descriptor_sha256"],
            "case_set_sha256": descriptor["case_set_sha256"],
            "dataset_sha256": multi._sha(output.read_bytes()), "record_count": len(records),
            "qualification_protocol": descriptor["qualification_protocol"],
            "case_coverage": coverage, "shard_manifests": shard_manifests}
        if family in ("ub", "p"):
            manifest.update({"source_dataset_sha256": source_binding["sha256"],
                "source_manifest_sha256": source_binding["manifest_sha256"],
                "case_source_bindings": [{"case_key": [case["scenario_id"], case["category"]],
                    "source_dataset_path": item["path"],
                    "source_dataset_sha256": item["sha256"],
                    "source_manifest_path": item["manifest_path"],
                    "source_manifest_sha256": item["manifest_sha256"]}
                    for case, item in zip(expected_cases, source_shards)]})
        sidecar = output.with_suffix(output.suffix + ".manifest.json")
        sidecar.write_text(json.dumps(manifest, sort_keys=True))
        result = {"path": str(output), "sha256": multi._sha(output.read_bytes()),
            "manifest_path": str(sidecar), "manifest_sha256": multi._sha(sidecar.read_bytes()),
            "record_count": len(records), "case_coverage": coverage}
        if family in ("ub", "p"):
            result.update({"source_dataset_sha256": source_binding["sha256"],
                "source_manifest_sha256": source_binding["manifest_sha256"]})
        if combine_hook:
            combine_hook(family, shards, output)
        return result

    def fit(combined, artifact_paths, *, descriptor, shard_receipts=None):
        for track, artifact in artifact_paths.items():
            artifact.write_bytes(track.encode())
        return {"descriptor_sha256": descriptor["descriptor_sha256"],
            "combined_inputs": combined}

    def validate(track, artifact, combined, *, descriptor):
        return {"track": track, "artifact_sha256": multi._sha(Path(artifact).read_bytes()),
            "descriptor_sha256": descriptor["descriptor_sha256"],
            "case_set_sha256": descriptor["case_set_sha256"], "combined_inputs": combined}

    from contextlib import contextmanager
    @contextmanager
    def lease(_devices, *, owner):
        events.append(("lease_enter", owner))
        try:
            yield {"synthetic": True}
        finally:
            events.append(("lease_exit", owner))

    def attest(_lease, devices):
        events.append(("attest", tuple(devices)))
        return {"logical_lease": {"devices": devices}}

    return collect, combine, fit, validate, lease, attest, collected, events


def _run_synthetic_execute(monkeypatch, tmp_path, descriptor, *, collector_hook=None,
        combine_hook=None):
    from frc_defense import meco_phase2_complete_set
    monkeypatch.setattr(multi, "validate_multicase_descriptor", lambda value: value)
    monkeypatch.setattr(meco_phase2_complete_set, "validate_complete_set_manifest",
        lambda manifest, _descriptor: manifest if manifest == {"synthetic_complete": True}
        else (_ for _ in ()).throw(ValueError("synthetic complete-set evidence mismatch")))
    adapters = _execute_adapters(tmp_path, descriptor,
        collector_hook=collector_hook, combine_hook=combine_hook)
    collect, combine, fit, validate, lease, attest, collected, events = adapters
    result = multi.execute_multicase(descriptor, collect_case=collect,
        combine_target=combine, fit_combined=fit, validate_track=validate,
        lease_factory=lease, device_identity_verifier=attest)
    return result, collected, events, adapters


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


def test_execute_multicase_publishes_only_after_serial_shards_and_five_receipts(
        tmp_path, frozen_protocol, monkeypatch):
    protocol, repo = frozen_protocol
    descriptor = _describe(tmp_path, protocol, repo, [
        {"scenario_id": "case-a", "category": "single"},
        {"scenario_id": "case-b", "category": "pair"}])
    result, collected, events, _ = _run_synthetic_execute(monkeypatch, tmp_path, descriptor)
    assert [case for case, _ in collected] == [["case-a", "single"], ["case-b", "pair"]]
    assert [event[0] for event in events] == ["lease_enter", "attest", "collect",
        "collect", "lease_exit"]
    assert [event[1] for event in events if event[0] == "collect"] == [
        ("case-a", "single"), ("case-b", "pair")]
    assert set(result["artifacts"]) == set(multi.MECO_TRACKS)
    assert result["phase2_ready"] is result["phase2_complete"] is False
    assert result["qualification_eligible"] is False
    assert result["target_coverage"]["trajectory"]["case-a::single"]["observed_seeds"] == [11, 12]
    assert result["target_coverage"]["p"]["case-b::pair"]["observed_seed_count"] == 2
    assert json.loads(Path(descriptor["outputs"]["manifest"]).read_text()) == result
    multi.validate_five_track_manifest(result, descriptor,
        validator_receipts=result["validator_receipts"],
        artifact_paths={track: Path(entry["path"]) for track, entry in result["artifacts"].items()})
    pooled = Path(result["combined_inputs"]["p"]["path"])
    pooled.write_bytes(pooled.read_bytes() + b"tamper\n")
    with pytest.raises(ValueError, match="combined p bytes changed"):
        multi.validate_five_track_manifest(result, descriptor,
            validator_receipts=result["validator_receipts"],
            artifact_paths={track: Path(entry["path"]) for track, entry in result["artifacts"].items()})


@pytest.mark.parametrize("tamper_kind", ["receipt", "malformed", "malformed_row",
    "trajectory", "ub", "p", "sidecar"])
def test_execute_resume_rejects_tampered_receipt_and_retries_fresh_attempt(
        tmp_path, frozen_protocol, monkeypatch, tamper_kind):
    protocol, repo = frozen_protocol
    descriptor = _describe(tmp_path, protocol, repo, [
        {"scenario_id": "case-a", "category": "single"},
        {"scenario_id": "case-b", "category": "pair"}])
    failed_once = {"value": False}
    def interrupt_second_case(case, _attempt):
        if case["case_key"] == ["case-b", "pair"] and not failed_once["value"]:
            failed_once["value"] = True
            raise RuntimeError("injected interruption")
    from frc_defense import meco_phase2_complete_set
    monkeypatch.setattr(multi, "validate_multicase_descriptor", lambda value: value)
    monkeypatch.setattr(meco_phase2_complete_set, "validate_complete_set_manifest",
        lambda manifest, _descriptor: manifest if manifest == {"synthetic_complete": True}
        else (_ for _ in ()).throw(ValueError("synthetic complete-set evidence mismatch")))
    adapters = _execute_adapters(tmp_path, descriptor, collector_hook=interrupt_second_case)
    collect, combine, fit, validate, lease, attest, collected, events = adapters
    with pytest.raises(RuntimeError, match="injected interruption"):
        multi.execute_multicase(descriptor, collect_case=collect, combine_target=combine,
            fit_combined=fit, validate_track=validate, lease_factory=lease,
            device_identity_verifier=attest)
    case_a = descriptor["outputs"]["root"] + "/shards/case-0001/attempt-0001"
    receipt_path = Path(case_a) / "case-receipt.json"
    if tamper_kind == "receipt":
        receipt = json.loads(receipt_path.read_text())
        receipt["case_key"] = ["forged", "single"]
        receipt_path.write_text(json.dumps(receipt))
    elif tamper_kind == "malformed":
        receipt = json.loads(receipt_path.read_text())
        receipt["datasets"]["trajectory"] = None
        receipt_path.write_text(json.dumps(receipt))
    elif tamper_kind == "malformed_row":
        receipt = json.loads(receipt_path.read_text())
        entry = receipt["datasets"]["trajectory"]
        data_path, manifest_path = Path(entry["path"]), Path(entry["manifest_path"])
        data_path.write_text("null\nnull\n")
        sidecar = json.loads(manifest_path.read_text())
        sidecar["dataset_sha256"] = multi._sha(data_path.read_bytes())
        manifest_path.write_text(json.dumps(sidecar, sort_keys=True))
        entry["sha256"] = multi._sha(data_path.read_bytes())
        entry["manifest_sha256"] = multi._sha(manifest_path.read_bytes())
        receipt_path.write_text(json.dumps(receipt))
    elif tamper_kind == "sidecar":
        receipt = json.loads(receipt_path.read_text())
        sidecar = Path(receipt["datasets"]["ub"]["manifest_path"])
        sidecar.write_text(sidecar.read_text() + " ")
    else:
        receipt = json.loads(receipt_path.read_text())
        source = Path(receipt["datasets"][tamper_kind]["path"])
        source.write_text(source.read_text() + "{}\n")
    result = multi.execute_multicase(descriptor, collect_case=collect,
        combine_target=combine, fit_combined=fit, validate_track=validate,
        lease_factory=lease, device_identity_verifier=attest)
    assert result["phase2_complete"] is False
    attempts = sorted(Path(descriptor["outputs"]["root"]
        + "/shards/case-0001").glob("attempt-*"))
    assert [item.name for item in attempts] == ["attempt-0001", "attempt-0002"]
    assert [case for case, _ in collected].count(["case-a", "single"]) == 2
    assert events.count(("lease_enter", "phase2-multicase-" + descriptor["descriptor_sha256"][:16])) == 2
    assert events.count(("lease_exit", "phase2-multicase-" + descriptor["descriptor_sha256"][:16])) == 2


def test_execute_resume_after_first_shard_interruption_uses_new_attempt(
        tmp_path, frozen_protocol, monkeypatch):
    protocol, repo = frozen_protocol
    descriptor = _describe(tmp_path, protocol, repo, [
        {"scenario_id": "case-a", "category": "single"},
        {"scenario_id": "case-b", "category": "pair"}])
    stopped = {"value": False}
    def interrupt_first_case(case, _attempt):
        if case["case_key"] == ["case-a", "single"] and not stopped["value"]:
            stopped["value"] = True
            raise RuntimeError("first-shard interruption")
    from frc_defense import meco_phase2_complete_set
    monkeypatch.setattr(multi, "validate_multicase_descriptor", lambda value: value)
    monkeypatch.setattr(meco_phase2_complete_set, "validate_complete_set_manifest",
        lambda manifest, _descriptor: manifest if manifest == {"synthetic_complete": True}
        else (_ for _ in ()).throw(ValueError("synthetic complete-set evidence mismatch")))
    collect, combine, fit, validate, lease, attest, collected, _events = _execute_adapters(
        tmp_path, descriptor, collector_hook=interrupt_first_case)
    with pytest.raises(RuntimeError, match="first-shard interruption"):
        multi.execute_multicase(descriptor, collect_case=collect, combine_target=combine,
            fit_combined=fit, validate_track=validate, lease_factory=lease,
            device_identity_verifier=attest)
    result = multi.execute_multicase(descriptor, collect_case=collect,
        combine_target=combine, fit_combined=fit, validate_track=validate,
        lease_factory=lease, device_identity_verifier=attest)
    attempts = sorted(Path(descriptor["outputs"]["root"]
        + "/shards/case-0001").glob("attempt-*"))
    assert [item.name for item in attempts] == ["attempt-0001", "attempt-0002"]
    assert not (attempts[0] / "case-receipt.json").exists()
    assert (attempts[1] / "case-receipt.json").is_file()
    assert [case for case, _ in collected] == [["case-a", "single"], ["case-b", "pair"]]
    assert set(result["artifacts"]) == set(multi.MECO_TRACKS)


def test_execute_resume_adopts_valid_first_shard_after_second_shard_interruption(
        tmp_path, frozen_protocol, monkeypatch):
    protocol, repo = frozen_protocol
    descriptor = _describe(tmp_path, protocol, repo, [
        {"scenario_id": "case-a", "category": "single"},
        {"scenario_id": "case-b", "category": "pair"}])
    stopped = {"value": False}
    def interrupt_second_case(case, _attempt):
        if case["case_key"] == ["case-b", "pair"] and not stopped["value"]:
            stopped["value"] = True
            raise RuntimeError("second-shard interruption")
    from frc_defense import meco_phase2_complete_set
    monkeypatch.setattr(multi, "validate_multicase_descriptor", lambda value: value)
    monkeypatch.setattr(meco_phase2_complete_set, "validate_complete_set_manifest",
        lambda manifest, _descriptor: manifest if manifest == {"synthetic_complete": True}
        else (_ for _ in ()).throw(ValueError("synthetic complete-set evidence mismatch")))
    collect, combine, fit, validate, lease, attest, collected, events = _execute_adapters(
        tmp_path, descriptor, collector_hook=interrupt_second_case)
    with pytest.raises(RuntimeError, match="second-shard interruption"):
        multi.execute_multicase(descriptor, collect_case=collect, combine_target=combine,
            fit_combined=fit, validate_track=validate, lease_factory=lease,
            device_identity_verifier=attest)
    result = multi.execute_multicase(descriptor, collect_case=collect,
        combine_target=combine, fit_combined=fit, validate_track=validate,
        lease_factory=lease, device_identity_verifier=attest)
    case_a_root = Path(descriptor["outputs"]["root"] + "/shards/case-0001")
    case_b_root = Path(descriptor["outputs"]["root"] + "/shards/case-0002")
    assert sorted(path.name for path in case_a_root.glob("attempt-*")) == ["attempt-0001"]
    assert sorted(path.name for path in case_b_root.glob("attempt-*")) == ["attempt-0001", "attempt-0002"]
    assert [case for case, _ in collected].count(["case-a", "single"]) == 1
    assert [case for case, _ in collected].count(["case-b", "pair"]) == 1
    assert len([event for event in events if event[0] == "collect" and event[1] == ("case-a", "single")]) == 1
    assert result["phase2_complete"] is False


def test_execute_resume_does_not_swallow_runtime_source_validation_failure(
        tmp_path, frozen_protocol, monkeypatch):
    protocol, repo = frozen_protocol
    descriptor = _describe(tmp_path, protocol, repo, [
        {"scenario_id": "case-a", "category": "single"},
        {"scenario_id": "case-b", "category": "pair"}])
    stopped = {"value": False}
    def interrupt_second_case(case, _attempt):
        if case["case_key"] == ["case-b", "pair"] and not stopped["value"]:
            stopped["value"] = True
            raise RuntimeError("injected interruption")
    from frc_defense import meco_phase2_complete_set
    monkeypatch.setattr(multi, "validate_multicase_descriptor", lambda value: value)
    monkeypatch.setattr(meco_phase2_complete_set, "validate_complete_set_manifest",
        lambda manifest, _descriptor: manifest if manifest == {"synthetic_complete": True}
        else (_ for _ in ()).throw(ValueError("synthetic complete-set evidence mismatch")))
    collect, combine, fit, validate, lease, attest, _collected, _events = _execute_adapters(
        tmp_path, descriptor, collector_hook=interrupt_second_case)
    with pytest.raises(RuntimeError, match="injected interruption"):
        multi.execute_multicase(descriptor, collect_case=collect, combine_target=combine,
            fit_combined=fit, validate_track=validate, lease_factory=lease,
            device_identity_verifier=attest)
    monkeypatch.setattr(multi, "validate_case_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("runtime/source validation failure")))
    with pytest.raises(RuntimeError, match="runtime/source validation failure"):
        multi.execute_multicase(descriptor, collect_case=collect, combine_target=combine,
            fit_combined=fit, validate_track=validate, lease_factory=lease,
            device_identity_verifier=attest)
    attempts = sorted(Path(descriptor["outputs"]["root"]
        + "/shards/case-0001").glob("attempt-*"))
    assert [item.name for item in attempts] == ["attempt-0001"]


@pytest.mark.parametrize("tamper_family", ["trajectory", "ub", "p"])
@pytest.mark.parametrize("tamper_kind", ["data", "manifest"])
def test_execute_combiner_digest_drift_fails_before_fitting_or_bundle_publication(
        tmp_path, frozen_protocol, monkeypatch, tamper_family, tamper_kind):
    protocol, repo = frozen_protocol
    descriptor = _describe(tmp_path, protocol, repo, [
        {"scenario_id": "case-a", "category": "single"},
        {"scenario_id": "case-b", "category": "pair"}])
    def tamper_after_write(family, _shards, output):
        if family == tamper_family:
            path = (output if tamper_kind == "data" else
                output.with_suffix(output.suffix + ".manifest.json"))
            path.write_text(path.read_text() + "tampered by injected adapter\n")
    with pytest.raises(ValueError, match=f"combined {tamper_family} corpus"):
        _run_synthetic_execute(monkeypatch, tmp_path, descriptor,
            combine_hook=tamper_after_write)
    assert not Path(descriptor["outputs"]["manifest"]).exists()


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


def test_composite_counterfactual_reader_namespaces_repeated_seed_by_case(tmp_path, monkeypatch):
    from frc_defense import meco_supervised
    cases = [{"scenario_id": "s1", "category": "single", "seeds": [7]},
        {"scenario_id": "s2", "category": "pair", "seeds": [7]}]
    paths = []
    for i in range(2):
        path = tmp_path / f"shard-{i}.jsonl"
        path.write_text("fixture\n")
        paths.append(path)
    common = {"schema": "meco-counterfactual-trajectory-v3",
        "target_schema": "meco-matched-prefix-gross-return-v3",
        "records_have_prefix_replay_parity": True, "source_runtime_device_bound": True,
        "state_digest_schema": "meco-full-environment-state-v2", "sources": {"s": 1},
        "source_identity_sha256": "x", "device": {"device": "cuda:0"},
        "native_extension_identity_schema": 1, "native_extension_identity": {"status": "present"},
        "artifact_identity_sha256": "a", "prediction_mode": "cv",
        "prediction_contract": {"mode": "cv"}, "candidate_boundary": {}, "candidate_k": 8,
        "plan_a_support_order": "order", "route_cost_points_per_second": 1,
        "discount_semantics": "discount", "continuation_policy": "teacher",
        "plan_c_target_schema": "meco-plan-c-matched-repeated-gross-return-v1",
        "plan_c_repeat_protocol": {"repeat_protocol_id": "repeat"},
        "plan_a_p_target": None, "plan_a_p_prior": 0., "plan_c_repeat_variance_note": "R=2",
        "max_macro_ticks": 5, "horizon_physics_ticks": 10, "gamma_per_macro": .9,
        "max_plan_c_descriptors_per_prefix": 2, "max_plan_a_pairs_per_prefix": 2}
    def read(path, **kwargs):
        index = paths.index(Path(path))
        case = cases[index]
        manifest = {**common, "scenario_id": case["scenario_id"],
            "category": case["category"], "seeds": case["seeds"],
            "seed_branch_summaries": {"7": {"case": case["scenario_id"]}}}
        row = {"scenario_id": case["scenario_id"], "category": case["category"],
            "seed": 7, "decision_index": 0}
        return [row], str(index) * 64, chr(97 + index) * 64, {}, "p" * 64, "q" * 64, manifest
    monkeypatch.setattr(meco_supervised, "_read_counterfactual_dataset", read)
    rows, aggregate, reference, manifests = meco_supervised._read_counterfactual_composite_dataset(
        [{"path": paths[0]}, {"path": paths[1]}], expected_cases=cases,
        qualification_protocol=tmp_path / "protocol.json", expected_sources={"s": 1})
    assert [(row["scenario_id"], row["seed"]) for row in rows] == [("s1", 7), ("s2", 7)]
    assert set(aggregate["case_coverage"]) == {"s1::single", "s2::pair"}
    assert len(aggregate["cases"]) == 2
    assert aggregate["cases"][0]["case_key"] == ["s1", "single"]
    assert reference["scenario_id"] == "s1"
    assert len(manifests) == 2


def test_composite_c_pair_counts_treat_unstable_labels_as_abstentions():
    from frc_defense import meco_supervised
    row = {"plan_c_counterfactuals": [{}, {}, {}],
        "plan_c_pairwise_labels": [{"stable_preference": True},
            {"stable_preference": False}, {"stable_preference": False}]}
    assert meco_supervised._count_composite_plan_c_pairs([row]) == (1, 2)
    row["plan_c_pairwise_labels"][1]["stable_preference"] = 0
    with pytest.raises(ValueError, match="boolean stable_preference"):
        meco_supervised._count_composite_plan_c_pairs([row])


def test_plan_a_composite_alignment_joins_case_seed_decision_and_separate_masks():
    import torch
    from types import SimpleNamespace
    from frc_defense import meco_supervised
    cases = [{"case_key": ["s1", "single"], "seeds": [1]},
        {"case_key": ["s2", "pair"], "seeds": [1]}]
    ub_by_case, p_by_case = [], []
    for case, mask in zip(cases, (torch.tensor([[1, 0], [1, 0]], dtype=torch.bool),
            torch.tensor([[1, 1], [1, 0]], dtype=torch.bool))):
        key = tuple(case["case_key"])
        row = {"scenario_id": key[0], "category": key[1], "seed": 1, "decision_index": 0}
        lookup = torch.where(mask, torch.zeros_like(mask, dtype=torch.long),
            torch.full_like(mask, -1, dtype=torch.long))
        ub_by_case.append([{"row": row, "graph": SimpleNamespace(lookup=lookup.unsqueeze(0))}])
        p_by_case.append([{"row": row, "legal_xy_mask": mask}])
    aligned, coverage = meco_supervised._align_plan_a_ubp_composite_samples(
        cases, ub_by_case, p_by_case)
    assert len(aligned) == 2
    assert [item["case_key"] for item in coverage] == [["s1", "single"], ["s2", "pair"]]
    p_by_case[1][0]["row"] = {**p_by_case[1][0]["row"], "decision_index": 1}
    with pytest.raises(ValueError, match="exact matching"):
        meco_supervised._align_plan_a_ubp_composite_samples(cases, ub_by_case, p_by_case)


def test_default_native_composite_fit_adapter_routes_all_five_tracks(tmp_path, monkeypatch):
    from frc_defense import meco_supervised
    descriptor = {"descriptor_sha256": "d" * 64, "case_set_sha256": "k" * 64,
        "ordered_cases": [{"scenario_id": "s", "category": "c", "seeds": [1]}],
        "qualification_protocol": {"path": str(tmp_path / "protocol.json")},
        "prediction_mode": "cv", "device": "cuda:0",
        "c_settings": {"updates": 2, "learning_rate": .001},
        "a_settings": {"updates": 3, "learning_rate": .002, "route_margin": .02,
            "map_regularization": .001, "minimum_coverage_fraction": .01}}
    monkeypatch.setattr(multi, "validate_multicase_descriptor", lambda value: value)
    monkeypatch.setattr("frc_defense.curriculum_training.source_identity", lambda: {"source": "x"})
    calls = []
    def fake_c(track, shards, output, **kwargs):
        calls.append((track, "C", shards, kwargs))
        Path(output).write_bytes(track.encode())
    def fake_a(track, *args, **kwargs):
        calls.append((track, "A", kwargs["composite_case_shards"], kwargs))
        Path(args[3]).write_bytes(track.encode())
    monkeypatch.setattr(meco_supervised, "warm_start_counterfactual_composite_v1", fake_c)
    monkeypatch.setattr(meco_supervised, "warm_start_plan_a_joint_ubp_v1", fake_a)
    monkeypatch.setattr(meco_supervised, "validate_warm_start_artifact",
        lambda track, path, **kwargs: {"track": track, "case_set_sha256": "k" * 64,
            "descriptor_sha256": "d" * 64})
    paths = {track: tmp_path / f"{track}.pt" for track in multi.MECO_TRACKS}
    combined = {name: {"path": f"/{name}", "sha256": name, "manifest_path": f"/{name}.m",
        "manifest_sha256": name + "m", "record_count": 1,
        "case_coverage": {"s::c": {"observed_seeds": [1]}},
        **({"source_dataset_sha256": "trajectory", "source_manifest_sha256": "trajectorym"}
            if name in ("ub", "p") else {})}
        for name in ("trajectory", "ub", "p")}
    receipt = {"case_key": ["s", "c"], "seeds": [1], "datasets": {
        family: {"path": f"/{family}"} for family in ("trajectory", "ub", "p")}}
    result = multi.fit_multicase_warm_starts(combined, paths, descriptor=descriptor,
        shard_receipts=[receipt])
    canonical = {family: {key: value[key] for key in
        ("path", "sha256", "manifest_path", "manifest_sha256")}
        for family, value in combined.items()}
    assert result["combined_inputs"] == canonical
    assert next(call for call in calls if call[1] == "C")[3]["combined_inputs"] == canonical["trajectory"]
    assert next(call for call in calls if call[1] == "A")[3]["composite_combined_inputs"] == canonical
    assert [call[0] for call in calls] == list(multi.MECO_TRACKS)
    assert [call[1] for call in calls] == ["A", "A", "A", "C", "C"]
    assert result["case_shard_bindings"][0]["case_key"] == ["s", "c"]
    validation = multi.validate_multicase_warm_start_track("A1", paths["A1"], combined,
        descriptor=descriptor)
    assert validation["combined_inputs"] == canonical
