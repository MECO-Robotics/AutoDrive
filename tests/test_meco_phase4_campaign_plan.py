import ast
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
from frc_defense.meco_phase4_opponents import POLICY_POOL, compile_opponent_policy_schedule
from frc_defense.meco_validation_campaign_runner import (
    run_phase4_opponent_cell, run_phase4_start_profile_cell)
import frc_defense.meco_validation_campaign_runner as campaign_runner
from frc_defense.meco_tracks import MECO_TRACKS


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "qualifications/meco/protocol.json"
REGISTRY = ROOT / "qualifications/meco/phase4_design_registry.json"
OVERLAY = ROOT / "qualifications/meco/phase4_design_overlay.json"


def test_curriculum_constructor_defers_initial_perception_until_profile_attachment():
    env_tree = ast.parse((ROOT / "frc_defense/tensor_3v3.py").read_text())
    env_class = next(node for node in env_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TensorThreeVsThreeEnv")
    constructor = next(node for node in env_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    reset_call = next(node for node in ast.walk(constructor)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reset")
    assert any(keyword.arg == "defer_perception" and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "defer_initial_perception" for keyword in reset_call.keywords)

    curriculum_tree = ast.parse((ROOT / "frc_defense/curriculum_scenarios.py").read_text())
    factory = next(node for node in curriculum_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_curriculum_env")
    env_call = next(node for node in ast.walk(factory)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "CurriculumEnv")
    assert any(keyword.arg == "defer_initial_perception"
        and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
        for keyword in env_call.keywords)
    assignment_attach = next(index for index, node in enumerate(factory.body)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr ==
            "phase4_start_profile_assignment" for target in node.targets))
    assert assignment_attach > next(index for index, node in enumerate(factory.body)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "env" for target in node.targets))
    scenario_reset = next(index for index, node in enumerate(factory.body)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "reset_curriculum_scenario")
    assert scenario_reset > assignment_attach


@pytest.mark.parametrize("field", ["checkpoint_sha256", "source_identity_sha256",
                                    "execution_device"])
def test_phase4_dispatch_binding_rejects_report_drift(field):
    expected = {"checkpoint_sha256": {"A1": "checkpoint"},
        "source_identity_sha256": "source", "execution_device": {"device": "cuda:1"}}
    observed = dict(expected)
    observed[field] = "drifted"
    with pytest.raises(ValueError, match="checkpoint/source/device binding mismatch"):
        campaign_runner._validate_phase4_dispatch_report_binding(
            observed, expected, context="test report")


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
    reactive = next(r for r in plan["requirements"]
                   if r["requirement_id"] == "reactive_defenders_and_opponent_pools")
    assert len(reactive["cells"]) == (
        len(reactive["design_axis_domains"]["policy_pool"])
        * reactive["applicable_protocol_case_count"])
    assert {cell["opponent_schedule_assignment"]["policy_id"]
            for cell in reactive["cells"]} == set(
                reactive["design_axis_domains"]["policy_pool"])
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
                if requirement["requirement_id"] == "reactive_defenders_and_opponent_pools":
                    assignment = cell["opponent_schedule_assignment"]
                    assert cell["scheduler_assignment_supported"] is True
                    assert cell["status"] == "protocol_bound_policy_assignment_planned_not_executed"
                    assert assignment["protocol_bound_evaluation_case"]["split"] == "validation"
                    assert assignment["policy_config_sha256"]
                    assert assignment["stimulus_identity_sha256"]
                    assert assignment["paired_control_category_seed_row_count"] == 1600
                    assert len(assignment["paired_control_category_seed_coverage_sha256"]) == 64
                    assert assignment["executed"] is False
                    assert assignment["qualification_eligible"] is False
                elif requirement["requirement_id"] == "multiple_legal_team_start_states":
                    assignment = cell["start_profile_assignment"]
                    assert cell["scheduler_assignment_supported"] is True
                    assert cell["status"] == (
                        "protocol_bound_start_profile_assignment_planned_not_executed")
                    assert assignment["registry_profile_id"] == cell["design_axes"][
                        "legal_team_pose_sets"]
                    assert assignment["protocol_id"] == plan["protocol_binding"]["protocol_id"]
                    assert assignment["protocol_sha256"] == plan["protocol_binding"][
                        "protocol_raw_file_sha256"]
                    assert assignment["outcome_by_start_state"] is None
                    assert assignment["qualification_eligible"] is False
                else:
                    assert cell["status"] == "design_only_unassigned_to_protocol_case"
                assert cell["executable"] is False
            else:
                # Protocol assignment alone does not make a cell executable:
                # the exact runtime stimulus and all required outcome signals
                # must both be wired before campaign dispatch.
                assert cell["executable"] is False
    assert "seeds" not in set(_walk_keys(plan))
    assert "final_test" not in set(_walk_keys(plan))


def test_phase4_reactive_cell_dispatch_threads_exact_schedule_to_matrix_cli(tmp_path):
    plan = compile_phase4_campaign_plan(PROTOCOL, REGISTRY, OVERLAY)
    requirement = next(r for r in plan["requirements"]
                       if r["requirement_id"] == "reactive_defenders_and_opponent_pools")
    cell = next(cell for cell in requirement["cells"]
        if cell["design_axes"].get("policy_pool") == "reactive_defender_a"
        and cell["case_axes"]["scenario_id"] == "offense_vs_scripted_pressure"
        and "level_4" in cell["case_axes"]["category"])
    checkpoints = {}
    for track in MECO_TRACKS:
        path = tmp_path / f"{track}.pt"
        path.write_bytes(track.encode())
        checkpoints[track] = str(path)
    observed = []

    def fake_runner(argv):
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(Path(argv[argv.index("--seed-manifest") + 1]).read_text())
        case_binding = json.loads(Path(argv[argv.index("--qualification-case-json") + 1]).read_text())
        if "--phase4-opponent-schedule-json" in argv:
            schedule_path = Path(argv[argv.index("--phase4-opponent-schedule-json") + 1])
            schedule = json.loads(schedule_path.read_text())
            assert schedule["policy_id"] == "reactive_defender_a"
            assert schedule["scenario_id"] == cell["case_axes"]["scenario_id"]
            assert schedule["categories"] == [cell["case_axes"]["category"]]
            assert schedule["split"] == "validation"
            assert schedule["qualification_eligible"] is False
            assert schedule["configured_pressure_slots"] == {
                cell["case_axes"]["category"]: [0]}
            arm = "factor_treatment"
        else:
            arm = "no_factor"
        assert output_dir.name == arm
        assert argv[argv.index("--conditions") + 1] == "architecture_matched_cv_k16"
        assert argv[argv.index("--seed-manifest") + 1]
        policies = (*MECO_TRACKS, "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2")
        rows = []
        for policy in policies:
            for category in manifest["categories"]:
                for seed in manifest["seed_list"]:
                    rows.append({"policy": policy, "category": category, "seed": seed,
                        "phase4_opponent_policy": (None if arm == "no_factor" else {
                            "policy_id": "reactive_defender_a",
                            "factor_id": schedule["factor_id"],
                            "policy_family": schedule["policy_family"],
                            "schedule_sha256": schedule["schedule_sha256"],
                            "interception_rate": None,
                            "paired_outcome_by_family": None})})
        report_path = output_dir / "condition.json"
        report_path.write_text(json.dumps({"qualification_eligible": False,
            "records": rows}))
        matrix_path = output_dir / "mock.matrix.json"
        matrix_path.write_text(json.dumps({"protocol": {
            "qualification_case": case_binding,
            "categories": manifest["categories"], "seed_list": manifest["seed_list"],
            "condition_selection": ["architecture_matched_cv_k16"],
            "conditions": [{"condition_name": "architecture_matched_cv_k16"}],
            "phase4_opponent_schedule": (schedule if arm == "factor_treatment" else None)},
            "conditions": {"architecture_matched_cv_k16": {"report_path": str(report_path)}}}))
        observed.append(argv)
        return 0

    result = run_phase4_opponent_cell(plan, cell["case_cell_key"],
        protocol_path=PROTOCOL, checkpoint_paths=checkpoints,
        matrix_script=tmp_path / "matrix.py", output_dir=tmp_path / "dispatch",
        device="cuda:1", runner=fake_runner)
    assert result["executed"] is True
    assert result["simulator_behavior_validated"] is False
    assert result["outcome_signals_complete"] is False
    assert result["dispatch_return_codes"] == {
        "no_factor": 0, "factor_treatment": 0}
    assert result["qualification_eligible"] is False
    assert result["required_family_outcomes"] == {
        "interception_rate": None, "policy_family_coverage": None,
        "paired_outcome_by_family": None}
    assert result["paired_coverage"]["exact_policy_category_seed_pair_coverage"] is True
    assert result["paired_coverage"]["qualification_eligible"] is False


def test_phase4_start_profile_dispatch_pairs_all_policies_and_exact_seeds(tmp_path, monkeypatch):
    plan = compile_phase4_campaign_plan(PROTOCOL, REGISTRY, OVERLAY)
    requirement = next(row for row in plan["requirements"]
                       if row["requirement_id"] == "multiple_legal_team_start_states")
    cell = next(row for row in requirement["cells"]
        if row["design_axes"].get("legal_team_pose_sets") == "team_start_set_b"
        and row["case_axes"]["scenario_id"] == "full_match_offense"
        and row["case_axes"]["start_condition"] == "offset_upper_lane")
    checkpoints = {}
    for track in MECO_TRACKS:
        path = tmp_path / f"{track}.pt"
        path.write_bytes(track.encode())
        checkpoints[track] = str(path)
    expected_binding = {"checkpoint_sha256": {
        track: campaign_runner._file_sha(Path(path)) for track, path in checkpoints.items()},
        "source_identity_sha256": "frozen-source-identity",
        "execution_device": {"device": "cuda:1", "accelerator_backend": "ROCm"}}
    monkeypatch.setattr(campaign_runner, "_capture_phase4_dispatch_binding",
        lambda checkpoint_paths, device: expected_binding)
    monkeypatch.setattr(campaign_runner, "_source_identity_sha256",
        lambda: expected_binding["source_identity_sha256"])
    observed = []

    def fake_runner(argv):
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(Path(argv[argv.index("--seed-manifest") + 1]).read_text())
        case_binding = json.loads(Path(argv[argv.index("--qualification-case-json") + 1]).read_text())
        has_profile = "--phase4-start-profile" in argv
        arm = "factor_treatment" if has_profile else "no_factor"
        assert output_dir.name == arm
        if has_profile:
            assert argv[argv.index("--phase4-start-profile") + 1] == "team_start_set_b"
        assert argv[argv.index("--conditions") + 1] == "architecture_matched_cv_k16"
        assert manifest["seed_list"] and len(manifest["seed_list"]) == cell["seed_count"]
        policies = (*MECO_TRACKS, "A1_P0", "A2_P0", "A3_P0", "D0", "D1", "D2")
        assignment = cell["start_profile_assignment"] if has_profile else None
        rows = []
        for policy in policies:
            for category in manifest["categories"]:
                for seed in manifest["seed_list"]:
                    rows.append({"policy": policy, "category": category, "seed": seed,
                        "checkpoint_sha256": (expected_binding["checkpoint_sha256"].get(
                            policy[:-3] if policy.endswith("_P0") else policy)),
                        "source_identity_sha256": expected_binding["source_identity_sha256"],
                        "execution_device": expected_binding["execution_device"],
                        "phase4_start_profile": (None if assignment is None else {
                            "factor_id": assignment["factor_id"],
                            "registry_profile_id": assignment["registry_profile_id"],
                            "runtime_profile_name": assignment["runtime_profile_name"],
                            "profile_config_sha256": assignment["profile_config_sha256"],
                            "assignment_sha256": assignment["assignment_sha256"],
                            "profile_applied_before_observation": True,
                            "legal_start_validation_rate": None,
                            "outcome_by_start_state": None,
                            "qualification_eligible": False})})
        report_path = output_dir / "condition.json"
        policy_manifest = {}
        for policy in policies:
            track = policy[:-3] if policy.endswith("_P0") else policy
            policy_manifest[policy] = {
                "checkpoint_sha256": expected_binding["checkpoint_sha256"].get(track),
                "source_identity_sha256": expected_binding["source_identity_sha256"],
                "execution_device": expected_binding["execution_device"]}
        report_path.write_text(json.dumps({"qualification_eligible": False,
            "checkpoint_sha256": expected_binding["checkpoint_sha256"],
            "source_identity_sha256": expected_binding["source_identity_sha256"],
            "execution_device": expected_binding["execution_device"],
            "policy_manifest": policy_manifest,
            "phase4_start_profile_assignment": assignment, "records": rows}))
        matrix_payload = {"qualification_eligible": False,
            "checkpoint_sha256": expected_binding["checkpoint_sha256"],
            "source_identity_sha256": expected_binding["source_identity_sha256"],
            "execution_device": expected_binding["execution_device"],
            "phase4_start_profile_assignment": assignment,
            "protocol": {"qualification_case": case_binding,
                "categories": manifest["categories"], "seed_list": manifest["seed_list"],
                "checkpoint_sha256": expected_binding["checkpoint_sha256"],
                "source_identity_sha256": expected_binding["source_identity_sha256"],
                "execution_device": expected_binding["execution_device"],
                "phase4_start_profile_assignment": assignment},
            "conditions": {"architecture_matched_cv_k16": {
                "checkpoint_sha256": expected_binding["checkpoint_sha256"],
                "source_identity_sha256": expected_binding["source_identity_sha256"],
                "execution_device": expected_binding["execution_device"],
                "report_path": str(report_path)}}}
        (output_dir / "mock.matrix.json").write_text(json.dumps(matrix_payload))
        observed.append(argv)
        return 0

    result = run_phase4_start_profile_cell(plan, cell["case_cell_key"],
        protocol_path=PROTOCOL, checkpoint_paths=checkpoints,
        matrix_script=tmp_path / "matrix.py", output_dir=tmp_path / "dispatch",
        device="cuda:1", runner=fake_runner)
    assert result["executed"] is True
    assert result["simulator_behavior_validated"] is False
    assert result["qualification_eligible"] is False
    assert result["legal_start_validation_rate"] is None
    assert result["outcome_by_start_state"] is None
    assert result["dispatch_return_codes"] == {"no_factor": 0, "factor_treatment": 0}
    assert result["paired_coverage"]["paired_row_count"] == (
        (len(MECO_TRACKS) + 3 + 3) * cell["seed_count"])
    assert len(observed) == 2
    assert all("--phase4-start-profile" not in argv for argv in observed[:1])
    assert "--phase4-start-profile" in observed[1]
    assert result["paired_coverage"]["paired_outcome_by_family"] is None
    assert result["paired_coverage"]["condition_names"] == ["architecture_matched_cv_k16"]
    assert result["paired_coverage"]["paired_rows_per_condition"][
        "architecture_matched_cv_k16"] == (
        11 * cell["seed_count"])
    assert len(observed) == 2


def test_all_policy_ids_match_plan_to_runtime_schedule_identity():
    plan = compile_phase4_campaign_plan(PROTOCOL, REGISTRY, OVERLAY)
    requirement = next(r for r in plan["requirements"]
                       if r["requirement_id"] == "reactive_defenders_and_opponent_pools")
    case = next(cell for cell in requirement["cells"]
        if cell["design_axes"].get("policy_pool") == POLICY_POOL[0]
        and "level_4" in cell["case_axes"]["category"])
    protocol = json.loads(PROTOCOL.read_text())
    case_axes = case["case_axes"]
    protocol_row = next(row for row in protocol["cases"]
        if all(row.get(axis) == case_axes[axis] for axis in case_axes))
    qualification_case = {axis: case_axes[axis] for axis in (
        "layout_id", "defender_family", "start_condition", "sensor_profile")}
    for policy_id in POLICY_POOL:
        planned = next(cell for cell in requirement["cells"]
            if cell["case_axes"] == case_axes
            and cell["design_axes"].get("policy_pool") == policy_id)
        assignment = planned["opponent_schedule_assignment"]
        schedule = compile_opponent_policy_schedule(policy_id=policy_id,
            scenario_id=case_axes["scenario_id"], categories=(case_axes["category"],),
            qualification_case=qualification_case, seeds=protocol_row["seeds"],
            split="validation", factor_id=planned["factor_id"])
        assert schedule["factor_id"] == planned["factor_id"] == "defender.reactive_pool"
        assert schedule["policy_family"] == assignment["policy_family"]
        assert schedule["per_episode_assignment_sha256"] == assignment[
            "factor_treatment_seeded_assignment_sha256"]
        assert schedule["configured_pressure_slots"][case_axes["category"]] == [0]


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
