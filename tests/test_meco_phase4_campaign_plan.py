import json
from pathlib import Path

import pytest

from frc_defense.meco_phase4_campaign_plan import (
    CONDITIONS,
    EXPECTED_FACTOR_CASE_AXIS_CELLS,
    POLICIES,
    compile_phase4_campaign_plan,
)
from frc_defense.meco_phase4_design import compile_phase4_design_overlay


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "qualifications/meco/protocol.json"
REGISTRY = ROOT / "qualifications/meco/phase4_design_registry.json"
OVERLAY = ROOT / "qualifications/meco/phase4_design_overlay.json"


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def test_phase4_plan_has_exact_compact_paired_schedule():
    plan = compile_phase4_campaign_plan(PROTOCOL, REGISTRY, OVERLAY)
    assert plan["factor_case_axis_cell_count"] == EXPECTED_FACTOR_CASE_AXIS_CELLS == 6488
    assert plan["scheduled_row_count_theoretical"] == 10_380_800
    assert plan["episode_rows_materialized"] is False
    assert plan["final_test_rows_or_seed_values_included"] is False
    assert plan["design_only_cells_executable"] is False
    assert plan["executable_cell_count"] == 0
    assert plan["runtime_capture_schema"] == "meco-phase4-realized-stimulus-capture-v1"
    assert plan["policies"] == list(POLICIES)
    assert plan["pair_conditions"] == list(CONDITIONS)
    assert sum(r["factor_case_axis_cell_count"] for r in plan["requirements"]) == 6488
    trench = next(r for r in plan["requirements"]
                  if r["requirement_id"] == "trench_interception")
    trench_categories = {cell["case_axes"]["category"] for cell in trench["cells"]}
    assert len(trench_categories) == 14
    assert all(category.split("/")[1] == "trench" for category in trench_categories)
    for requirement in plan["requirements"]:
        assert requirement["pairing_grid"] == [
            {"policy_id": policy, "condition": condition}
            for policy in POLICIES for condition in CONDITIONS]
        assert requirement["pair_coverage"].startswith("exactly_one_control")
        assert requirement["runtime_binding"]["required_signals"]
        assert requirement["runtime_binding"]["factor_ids"]
        for cell in requirement["cells"]:
            assert cell["seed_count"] == 100
            assert cell["planned_pair_rows_per_policy"] == 200
            assert cell["executed"] is False
            expected_runtime_factor = requirement["runtime_binding"][
                "runtime_factor_by_factor_id"].get(cell["factor_id"])
            assert cell["runtime_factor_kind"] == expected_runtime_factor
            assert cell["runtime_event"] == requirement["runtime_binding"][
                "runtime_event_by_factor_id"].get(cell["factor_id"])
            assert cell["runtime_stimulus_supported"] == bool(expected_runtime_factor)
            assert cell["case_cell_key"]
            if cell["design_axes"]:
                assert cell["status"] == "design_only_unassigned_to_protocol_case"
                assert cell["executable"] is False
            else:
                # Protocol assignment alone does not make a cell executable:
                # the exact runtime stimulus and all required outcome signals
                # must both be wired before campaign dispatch.
                assert cell["executable"] is False
    assert "seeds" not in set(_walk_keys(plan))
    assert "final_test" not in set(_walk_keys(plan))


def test_rejects_protocol_registry_or_overlay_digest_mismatch(tmp_path):
    protocol = tmp_path / "protocol.json"
    registry = tmp_path / "registry.json"
    overlay = tmp_path / "overlay.json"
    protocol.write_bytes(PROTOCOL.read_bytes())
    registry.write_bytes(REGISTRY.read_bytes())
    overlay.write_bytes(OVERLAY.read_bytes())
    registry.write_text(REGISTRY.read_text() + "\n")
    with pytest.raises(ValueError, match="overlay registry raw digest mismatch|exactly match"):
        compile_phase4_campaign_plan(protocol, registry, overlay)


def test_rejects_unpaired_or_missing_policy_contract(tmp_path):
    registry_value = json.loads(REGISTRY.read_text())
    registry_value["pairing"]["same_checkpoint_controls"].pop()
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(registry_value))
    with pytest.raises(ValueError, match="exactly match"):
        compile_phase4_campaign_plan(PROTOCOL, registry, OVERLAY)


def test_rejects_case_reassigned_out_of_validation_split(tmp_path):
    protocol_value = json.loads(PROTOCOL.read_text())
    validation_row = next(row for row in protocol_value["cases"]
                          if row["split"] == "validation")
    validation_row["split"] = "test"
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(protocol_value))
    with pytest.raises(ValueError, match="exactly match"):
        compile_phase4_campaign_plan(protocol, REGISTRY, OVERLAY)


def test_rejects_modified_overlay_even_if_case_rows_look_valid(tmp_path):
    overlay_value = json.loads(OVERLAY.read_text())
    overlay_value["requirements"][0]["protocol_bound_case_count_per_factor"] += 1
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps(overlay_value))
    with pytest.raises(ValueError, match="exactly match"):
        compile_phase4_campaign_plan(PROTOCOL, REGISTRY, overlay)


def test_campaign_reader_never_decodes_or_serializes_final_test_sentinel(tmp_path, monkeypatch):
    protocol_value = json.loads(PROTOCOL.read_text())
    sentinel = "FINAL_TEST_SENTINEL_817263549"
    protocol_value["final_test"]["seed_sentinel"] = sentinel
    test_case = next(row for row in protocol_value["cases"] if row["split"] == "test")
    test_case["seeds"] = [sentinel]
    protocol_path = tmp_path / "protocol.json"
    registry_path = tmp_path / "registry.json"
    overlay_path = tmp_path / "overlay.json"
    protocol_path.write_text(json.dumps(protocol_value))
    registry_path.write_bytes(REGISTRY.read_bytes())
    overlay = compile_phase4_design_overlay(protocol_path, registry_path)
    overlay_path.write_text(json.dumps(overlay))

    import frc_defense.meco_phase4_design as design_module
    original_loads = design_module.json.loads
    decoded_inputs = []

    def recording_loads(value, *args, **kwargs):
        decoded_inputs.append(value)
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(design_module.json, "loads", recording_loads)
    plan = compile_phase4_campaign_plan(protocol_path, registry_path, overlay_path)
    assert all(sentinel not in (value.decode("utf-8") if isinstance(value, bytes) else value)
               for value in decoded_inputs)
    assert sentinel not in json.dumps(plan, sort_keys=True)


@pytest.mark.parametrize("split_edit", ["unknown", "missing"])
def test_campaign_rejects_malformed_split_after_overlay_regeneration_attempt(tmp_path, split_edit):
    protocol_value = json.loads(PROTOCOL.read_text())
    train_case = next(row for row in protocol_value["cases"] if row["split"] == "train")
    if split_edit == "unknown":
        train_case["split"] = "holdout"
    else:
        del train_case["split"]
    protocol_path = tmp_path / "protocol.json"
    registry_path = tmp_path / "registry.json"
    protocol_path.write_text(json.dumps(protocol_value))
    registry_path.write_bytes(REGISTRY.read_bytes())

    expected_message = ("unsupported split metadata" if split_edit == "unknown"
                        else "missing split metadata")
    # A malformed protocol cannot produce a fresh overlay; verify that the
    # overlay generation step itself fails closed, then exercise the campaign
    # entry point with the previous overlay to ensure it independently fails
    # before accepting mismatched coverage.
    with pytest.raises(ValueError, match=expected_message):
        compile_phase4_design_overlay(protocol_path, registry_path)
    with pytest.raises(ValueError, match=expected_message):
        compile_phase4_campaign_plan(protocol_path, registry_path, OVERLAY)
