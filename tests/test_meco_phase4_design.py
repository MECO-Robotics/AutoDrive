import hashlib
import json
from pathlib import Path

import pytest

from frc_defense.meco_phase4_design import compile_phase4_design_overlay


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _write_fixture(tmp_path, *, overlap=False):
    protocol = {
        "protocol_version": "fixture-v1",
        "scenario_ids": ["offense_vs_scripted_pressure", "full_match_offense",
                         "team_delivery_cycles"],
        "training": {"defender_families": ["train-family"]},
        "validation": {"defender_families": (["train-family", "held-family"]
                                                  if overlap else ["held-family", "held-family-b"]),
                       "layouts": ["val-layout-a", "val-layout-b"]},
        "start_conditions": {"train": ["train-start"], "validation": ["val-start"],
                             "test": ["test-start"]},
        "sensor_profiles": {"train": ["standard"], "validation": ["noisy"],
                             "test": ["test-sensor"]},
        "final_test": {"defender_families": ["never-export-test-family"]},
        "cases": [],
    }
    pressure_categories = ["active/trench/level_3_trench", "inactive/trench/level_0",
                           "active/trenched/level_3_trench",
                           "active/not_trench/level_3_trench"]
    categories = {"offense_vs_scripted_pressure": pressure_categories,
                  "full_match_offense": ["full_match_single"],
                  "team_delivery_cycles": ["delivery_single"]}
    for scenario, scenario_categories in categories.items():
        for category in scenario_categories:
            for layout in protocol["validation"]["layouts"]:
                for family in protocol["validation"]["defender_families"]:
                    protocol["cases"].append({
                        "split": "validation", "scenario_id": scenario,
                        "category": category, "layout_id": layout,
                        "defender_family": family, "start_condition": "val-start",
                        "sensor_profile": "noisy", "seeds": [987654321]})
    protocol["cases"].extend([
        {"split": "train", "scenario_id": "offense_vs_scripted_pressure",
         "category": "level_1", "layout_id": "train-layout",
         "defender_family": "train-family", "start_condition": "train-start",
         "sensor_profile": "standard", "seeds": [123456789]},
        {"split": "test", "scenario_id": "offense_vs_scripted_pressure",
         "category": "level_1", "layout_id": "test-layout",
         "defender_family": "never-export-test-family", "start_condition": "test-start",
         "sensor_profile": "test-sensor", "seeds": [
             777777777777777777, 888888888888888888]},
    ])
    protocol["protocol_id"] = _sha(protocol)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    source_registry_path = (Path(__file__).resolve().parents[1] / "qualifications" / "meco" /
                            "phase4_design_registry.json")
    registry = json.loads(source_registry_path.read_text())
    trench = next(item for item in registry["requirements"]
                  if item["requirement_id"] == "trench_interception")
    trench["validation_case_axes"]["category"]["expected_values"] = pressure_categories[:2]
    trench["resolved_category_domain"]["expected_values"] = pressure_categories[:2]
    trench["resolved_category_domain"]["expected_category_count"] = 2
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    return protocol_path, registry_path


def test_design_overlay_is_deterministic_bound_and_does_not_disclose_test_seeds(tmp_path):
    protocol_path, registry_path = _write_fixture(tmp_path)
    first = compile_phase4_design_overlay(protocol_path, registry_path)
    second = compile_phase4_design_overlay(protocol_path, registry_path)
    assert first == second
    raw = protocol_path.read_bytes()
    assert first["protocol_binding"]["protocol_raw_file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert first["protocol_binding"]["protocol_id"] == json.loads(raw)["protocol_id"]
    assert first["protocol_binding"]["binding_method"].startswith(
        "protocol_id_metadata_plus_opaque_full_file_sha256")
    serialized = json.dumps(first, sort_keys=True)
    assert "777777777777777777" not in serialized
    assert "888888888888888888" not in serialized
    assert "never-export-test-family" not in serialized
    assert first["disclosure"]["contains_seed_values"] is False
    assert first["disclosure"]["contains_final_test_rows"] is False
    assert first["qualification_eligible"] is False
    assert first["family_separation"]["overlap"] == []
    assert first["family_separation"]["whole_family_holdout_verified"] is True
    assert {req["requirement_id"] for req in first["requirements"]} == {
        "stationary_blockage", "direct_pursuit", "trench_interception", "feints",
        "reactive_defenders_and_opponent_pools", "multiple_legal_team_start_states",
        "teammate_behaviors", "sensor_delay_and_low_confidence",
        "abrupt_opponent_change", "bumps_and_contact",
        "observation_only_focal_pose_mislocalization"}
    assert first["factor_registry_binding"]["registry_raw_file_sha256"] == hashlib.sha256(
        registry_path.read_bytes()).hexdigest()
    assert all(req["planned_paired_protocol_condition_rows_per_factor"] ==
               2 * req["protocol_bound_case_count_per_factor"]
               for req in first["requirements"])


def test_final_test_seed_sentinel_is_never_decoded_or_serialized(tmp_path, monkeypatch):
    protocol_path, registry_path = _write_fixture(tmp_path)
    sentinel = "FINAL_TEST_SENTINEL_817263549"
    protocol = json.loads(protocol_path.read_text())
    protocol["final_test"]["secret_seed"] = sentinel
    protocol["cases"][-1]["seeds"] = [sentinel]
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    import frc_defense.meco_phase4_design as design_module
    original_loads = design_module.json.loads
    decoded_inputs = []

    def recording_loads(value, *args, **kwargs):
        decoded_inputs.append(value)
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(design_module.json, "loads", recording_loads)
    overlay = compile_phase4_design_overlay(protocol_path, registry_path)
    assert all(sentinel not in (value.decode("utf-8") if isinstance(value, bytes) else value)
               for value in decoded_inputs)
    assert sentinel not in json.dumps(overlay, sort_keys=True)


@pytest.mark.parametrize("split_edit", ["unknown", "missing"])
def test_protocol_reader_rejects_unknown_or_missing_case_split(tmp_path, split_edit):
    protocol_path, registry_path = _write_fixture(tmp_path)
    protocol = json.loads(protocol_path.read_text())
    row = next(case for case in protocol["cases"] if case["split"] == "train")
    if split_edit == "unknown":
        row["split"] = "holdout"
    else:
        del row["split"]
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    expected_message = ("unsupported split metadata" if split_edit == "unknown"
                        else "missing split metadata")
    with pytest.raises(ValueError, match=expected_message):
        compile_phase4_design_overlay(protocol_path, registry_path)


def test_design_overlay_rejects_any_train_validation_family_overlap(tmp_path):
    protocol_path, registry_path = _write_fixture(tmp_path, overlap=True)
    with pytest.raises(ValueError, match="whole defender families overlap"):
        compile_phase4_design_overlay(protocol_path, registry_path)


def test_design_overlay_changes_binding_when_raw_protocol_changes(tmp_path):
    protocol_path, registry_path = _write_fixture(tmp_path)
    first = compile_phase4_design_overlay(protocol_path, registry_path)
    protocol = json.loads(protocol_path.read_text())
    protocol["display_note"] = "raw-byte-bound change"
    protocol["protocol_id"] = _sha({k: v for k, v in protocol.items()
                                    if k != "protocol_id"})
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    second = compile_phase4_design_overlay(protocol_path, registry_path)
    assert first["protocol_binding"]["protocol_raw_file_sha256"] != second["protocol_binding"]["protocol_raw_file_sha256"]
    assert first["overlay_id"] != second["overlay_id"]


def test_factor_rows_resolve_exact_protocol_axes_and_design_cross_products(tmp_path):
    protocol_path, registry_path = _write_fixture(tmp_path)
    overlay = compile_phase4_design_overlay(protocol_path, registry_path)
    by_id = {item["requirement_id"]: item for item in overlay["requirements"]}

    def cases(requirement_id):
        return {tuple(row[axis] for axis in (
            "scenario_id", "category", "layout_id", "defender_family",
            "start_condition", "sensor_profile"))
                for row in by_id[requirement_id]["protocol_bound_case_rows"]}

    pressure = {
        ("offense_vs_scripted_pressure", category, layout, family, "val-start", "noisy")
        for category in ("active/trench/level_3_trench", "inactive/trench/level_0",
                         "active/trenched/level_3_trench", "active/not_trench/level_3_trench")
        for layout in ("val-layout-a", "val-layout-b")
        for family in ("held-family", "held-family-b")}
    for requirement_id in ("stationary_blockage", "direct_pursuit", "feints",
                           "abrupt_opponent_change"):
        assert cases(requirement_id) == pressure
        assert by_id[requirement_id]["protocol_bound_case_count_per_factor"] == 16
    trench_expected = {
        ("offense_vs_scripted_pressure", category, layout, family, "val-start", "noisy")
        for category in ("active/trench/level_3_trench", "inactive/trench/level_0")
        for layout in ("val-layout-a", "val-layout-b")
        for family in ("held-family", "held-family-b")}
    assert cases("trench_interception") == trench_expected
    assert by_id["trench_interception"]["protocol_bound_case_count_per_factor"] == 8
    selector_detail = by_id["trench_interception"]["resolved_selector_details"][0]
    assert selector_detail["resolved_values"] == [
        "active/trench/level_3_trench", "inactive/trench/level_0"]

    reactive_expected = pressure | {
        ("full_match_offense", "full_match_single", layout, family, "val-start", "noisy")
        for layout in ("val-layout-a", "val-layout-b")
        for family in ("held-family", "held-family-b")}
    assert cases("reactive_defenders_and_opponent_pools") == reactive_expected
    assert by_id["reactive_defenders_and_opponent_pools"]["protocol_bound_case_count_per_factor"] == 20
    assert {row["design_axes"]["opponent_policy_pool"] for row in
            by_id["reactive_defenders_and_opponent_pools"]["design_only_rows"]} == {
        "stationary_blockage", "direct_pursuit", "trench_interceptor",
        "telegraph_reversal", "reactive_defender_a", "reactive_defender_b",
        "orbit_to_pursuit_switch"}
    assert by_id["reactive_defenders_and_opponent_pools"]["design_only_row_count_per_factor"] == 7

    teammate_expected = {
        (scenario, category, layout, family, "val-start", "noisy")
        for scenario, category in (("full_match_offense", "full_match_single"),
                                   ("team_delivery_cycles", "delivery_single"))
        for layout in ("val-layout-a", "val-layout-b")
        for family in ("held-family", "held-family-b")}
    assert cases("teammate_behaviors") == teammate_expected
    teammate_rows = by_id["teammate_behaviors"]["design_only_rows"]
    assert {row["design_axes"]["teammate_behavior"] for row in teammate_rows} == {
        "collect", "score", "ferry", "defend", "hold_position"}
    assert all(not row["counts_as_protocol_coverage"] and
               not row["counts_as_executed_coverage"] for row in teammate_rows)

    assert cases("multiple_legal_team_start_states") == {
        ("full_match_offense", "full_match_single", layout, family, "val-start", "noisy")
        for layout in ("val-layout-a", "val-layout-b")
        for family in ("held-family", "held-family-b")}
    assert cases("sensor_delay_and_low_confidence") == {
        (scenario, category, layout, family, "val-start", "noisy")
        for scenario, category in (("offense_vs_scripted_pressure", "active/trench/level_3_trench"),
                                   ("offense_vs_scripted_pressure", "inactive/trench/level_0"),
                                   ("offense_vs_scripted_pressure", "active/trenched/level_3_trench"),
                                   ("offense_vs_scripted_pressure", "active/not_trench/level_3_trench"),
                                   ("full_match_offense", "full_match_single"),
                                   ("team_delivery_cycles", "delivery_single"))
        for layout in ("val-layout-a", "val-layout-b")
        for family in ("held-family", "held-family-b")}
    assert "field bump terrain" in by_id["bumps_and_contact"]["contact_semantics"]["excluded_event"]
    assert by_id["bumps_and_contact"]["contact_semantics"]["thresholds_or_margins"] == "none declared; no minimum contact impulse, distance, duration, or recovery-performance threshold is implied"
    assert cases("bumps_and_contact") == {
        (scenario, category, layout, family, "val-start", "noisy")
        for scenario, category in (("offense_vs_scripted_pressure", "active/trench/level_3_trench"),
                                   ("offense_vs_scripted_pressure", "inactive/trench/level_0"),
                                   ("offense_vs_scripted_pressure", "active/trenched/level_3_trench"),
                                   ("offense_vs_scripted_pressure", "active/not_trench/level_3_trench"),
                                   ("full_match_offense", "full_match_single"))
        for layout in ("val-layout-a", "val-layout-b")
        for family in ("held-family", "held-family-b")}
    assert cases("observation_only_focal_pose_mislocalization") == {
        ("full_match_offense", "full_match_single", layout, family, "val-start", "noisy")
        for layout in ("val-layout-a", "val-layout-b")
        for family in ("held-family", "held-family-b")}

    assert {row["design_axes"]["team_pose_set"] for row in
            by_id["multiple_legal_team_start_states"]["design_only_rows"]} == {
        "team_start_set_a", "team_start_set_b", "team_start_set_c"}
    pose_rows = by_id["observation_only_focal_pose_mislocalization"]["design_only_rows"]
    assert {tuple(row["design_axes"][axis] for axis in ("pose_offset", "truth_pose"))
            for row in pose_rows} == {
        (offset, "simulator_truth_only_not_policy_input")
        for offset in ("translation_small", "translation_large", "heading_offset")}


def test_registry_rejects_unresolved_axis_and_incomplete_source_plan_ids(tmp_path):
    protocol_path, registry_path = _write_fixture(tmp_path)
    registry = json.loads(registry_path.read_text())
    registry["requirements"][0]["validation_case_axes"]["layout_id"] = "validation.unknown_axis"
    bad_path = tmp_path / "unknown-axis-registry.json"
    bad_path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="unknown protocol-axis selector"):
        compile_phase4_design_overlay(protocol_path, bad_path)

    registry = json.loads(registry_path.read_text())
    registry["requirements"].pop()
    bad_path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="differ from source plan"):
        compile_phase4_design_overlay(protocol_path, bad_path)


def test_real_protocol_resolves_exactly_fourteen_trench_route_categories():
    root = Path(__file__).resolve().parents[1]
    overlay = compile_phase4_design_overlay(
        root / "qualifications" / "meco" / "protocol.json",
        root / "qualifications" / "meco" / "phase4_design_registry.json")
    trench = next(req for req in overlay["requirements"]
                  if req["requirement_id"] == "trench_interception")
    resolved = trench["resolved_selector_details"][0]
    assert resolved["route_segment_index"] == 1
    assert resolved["equals"] == "trench"
    assert len(resolved["resolved_values"]) == 14
    assert all(category.split("/")[1] == "trench"
               for category in resolved["resolved_values"])
    assert all(len(category.split("/")) == 3 and "level_" in category.split("/")[2]
               for category in resolved["resolved_values"])
    assert trench["protocol_bound_case_count_per_factor"] == 56


def test_exact_category_selector_fails_closed_on_empty_or_wrong_domain(tmp_path):
    protocol_path, registry_path = _write_fixture(tmp_path)
    registry = json.loads(registry_path.read_text())
    trench = next(req for req in registry["requirements"]
                  if req["requirement_id"] == "trench_interception")
    trench["validation_case_axes"]["category"]["expected_values"] = [
        "active/trench/level_3_trench", "active/trenched/level_3_trench"]
    wrong_path = tmp_path / "wrong-domain-registry.json"
    wrong_path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="outside its declared domain"):
        compile_phase4_design_overlay(protocol_path, wrong_path)

    protocol = json.loads(protocol_path.read_text())
    for row in protocol["cases"]:
        if row.get("split") == "validation" and row.get("scenario_id") == "offense_vs_scripted_pressure":
            row["category"] = "active/trenched/level_3_trench"
    protocol["protocol_id"] = _sha({k: v for k, v in protocol.items()
                                    if k != "protocol_id"})
    protocol_path.write_text(json.dumps(protocol))
    trench["validation_case_axes"]["category"]["expected_values"] = [
        "active/trench/level_3_trench", "inactive/trench/level_0"]
    empty_path = tmp_path / "empty-registry.json"
    empty_path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="resolved to an empty set"):
        compile_phase4_design_overlay(protocol_path, empty_path)


def test_exact_category_overlay_contract_rejects_forged_domain_metadata(tmp_path):
    protocol_path, registry_path = _write_fixture(tmp_path)
    registry = json.loads(registry_path.read_text())
    trench = next(req for req in registry["requirements"]
                  if req["requirement_id"] == "trench_interception")
    trench["resolved_category_domain"]["expected_category_count"] = 14
    bad_path = tmp_path / "forged-domain-registry.json"
    bad_path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="inconsistent resolved category-domain metadata"):
        compile_phase4_design_overlay(protocol_path, bad_path)
