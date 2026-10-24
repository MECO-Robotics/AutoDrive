"""Dashboard curriculum payload contract; saved fixtures only, no simulation."""
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

from frc_defense.dashboard import create_handler
from frc_defense.dashboard_api import curriculum_status_payload, _category_passes, _curriculum_report_root
from frc_defense.training_curriculum import CURRICULUM_VERSION, SCENARIOS


def _run(root, *, qualified=(), diagnostic=False, sources=None):
    run = root / "runs" / "one"
    run.mkdir(parents=True)
    categories = []
    records = []
    for name in SCENARIOS["pickup_simple"].required_categories:
        categories.append({"category": name, "episodes": 100, "successes": 95,
                           "useful_throughput": 1.0, "reference_throughput": 1.0,
                           "paired": True, "reference_qualified": True,
                           "reference_protocol_id": "fixture-v1", "seeds": list(range(100)),
                           "reference_seeds": list(range(100))})
        records.append({"scenario_id": "pickup_simple", "category": name,
                        "reference_successes": 95})
    stats = {"sources": sources or {"sim": "fixture-digest"}, "updates": 7,
             "algorithm": "ppo", "architecture": "fixture",
             "config": {"eval_every": 1},
             "curriculum_state": {"qualified": list(qualified), "consecutive_passes": {"pickup_simple": 3},
                                  "consecutive_protocol": {"pickup_simple": "fixture-v1"},
                                  "last_qualified_checkpoint": {name: "policy" for name in qualified},
                                  "qualified_protocol": {name: "fixture-v1" for name in qualified}},
             "stats": [{"update": 7, "scenario_id": "pickup_simple"}]}
    (run / "stats.json").write_text(json.dumps(stats))
    for update in (5, 6, 7):
        report = {"protocol_id": "fixture-v1", "sources": stats["sources"],
                  "curriculum_version": CURRICULUM_VERSION, "evaluation_id": f"eval-{update}", "policy_id": "policy",
                  "split": "validation", "diagnostic": diagnostic,
                  "evaluation_results": [{"scenario_id": "pickup_simple", "scenario_version": "1",
                                          "protocol_id": "fixture-v1", "categories": categories}],
                  "records": records}
        (run / f"evaluation-{update}.json").write_text(json.dumps(report))
    return run


def _payload(root):
    return json.loads(curriculum_status_payload(root, curriculum_root=root, project_root=root))


def _http_payload(run_dir, *, curriculum_root=None, meco_data_root=None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(
        run_dir, curriculum_root=curriculum_root, meco_data_root=meco_data_root))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        return json.loads(urlopen(
            f"http://127.0.0.1:{server.server_port}/api/curriculum-status").read())
    finally:
        server.shutdown()
        server.server_close()


def test_external_curriculum_root_is_allowed_under_configured_meco_data_root(tmp_path):
    data_root = tmp_path / "external-data"
    curriculum_root = data_root / "evaluations" / "curriculum"
    run_dir = data_root / "checkpoints" / "defense"
    _run(curriculum_root)
    run_dir.mkdir(parents=True)
    checkout = Path(__file__).resolve().parents[1]

    api = json.loads(curriculum_status_payload(
        run_dir, curriculum_root=curriculum_root, project_root=checkout,
        meco_data_root=data_root))
    http = _http_payload(run_dir, curriculum_root=curriculum_root,
                         meco_data_root=data_root)

    assert len(api["runs"]) == 1
    assert len(http["runs"]) == 1


def test_explicit_curriculum_root_outside_checkout_and_configured_data_root_is_rejected(tmp_path):
    data_root = tmp_path / "configured-data"
    outside_root = tmp_path / "outside-data" / "curriculum"
    run_dir = data_root / "checkpoints" / "defense"
    _run(outside_root)
    run_dir.mkdir(parents=True)
    checkout = Path(__file__).resolve().parents[1]

    api = json.loads(curriculum_status_payload(
        run_dir, curriculum_root=outside_root, project_root=checkout,
        meco_data_root=data_root))
    http = _http_payload(run_dir, curriculum_root=outside_root,
                         meco_data_root=data_root)

    assert api["runs"] == []
    assert http["runs"] == []


def test_explicit_external_curriculum_root_through_symlink_is_rejected(tmp_path):
    data_root = tmp_path / "external-data"
    curriculum_root = data_root / "evaluations" / "curriculum"
    symlinked_root = data_root / "evaluations" / "curriculum-link"
    run_dir = data_root / "checkpoints" / "defense"
    _run(curriculum_root)
    symlinked_root.symlink_to(curriculum_root, target_is_directory=True)
    run_dir.mkdir(parents=True)

    api = json.loads(curriculum_status_payload(
        run_dir, curriculum_root=symlinked_root,
        project_root=Path(__file__).resolve().parents[1], meco_data_root=data_root))
    http = _http_payload(run_dir, curriculum_root=symlinked_root,
                         meco_data_root=data_root)

    assert api["runs"] == []
    assert http["runs"] == []


def test_implicit_curriculum_root_stays_confined_to_dashboard_checkout(tmp_path):
    data_root = tmp_path / "external-data"
    _run(data_root)
    checkout = Path(__file__).resolve().parents[1]

    api = json.loads(curriculum_status_payload(
        data_root, project_root=checkout, meco_data_root=data_root))
    http = _http_payload(data_root, meco_data_root=data_root)

    assert api["runs"] == []
    assert http["runs"] == []


def test_empty_curriculum_has_registry_and_no_runs(tmp_path):
    data = _payload(tmp_path)
    assert data["runs"] == []
    assert data["scenarios"]["pickup_simple"]["required_categories"] == ["single", "cluster"]


def test_dashboard_requires_explicit_curriculum_root_when_run_dir_is_not_one(tmp_path):
    defense_root = tmp_path / "checkpoints" / "rebuilt-gamepiece-defense"
    defense_root.mkdir(parents=True)
    curriculum_root = tmp_path / "evaluations" / "curriculum-campaign"
    _run(curriculum_root)
    assert _curriculum_report_root(defense_root, tmp_path) is None
    assert _curriculum_report_root(defense_root, tmp_path, curriculum_root) == curriculum_root


def test_queued_only_lane_is_pending_and_stale_queue_is_reported(tmp_path):
    lane = {"architecture": "fixture", "directory": "runs/queued", "status": "queued",
            "current": "pickup_simple", "completed_updates": 0}
    queue = tmp_path / "queue-status.json"
    queue.write_text(json.dumps({"lanes": [lane]}))
    data = _payload(tmp_path)
    assert len(data["runs"]) == 1
    assert data["runs"][0]["status"] == "queued"
    assert data["runs"][0]["current"] == "pickup_simple"
    import os, time
    os.utime(queue, (time.time() - 200, time.time() - 200))
    assert _payload(tmp_path)["runs"][0]["status"] == "stale"


def test_only_valid_paired_nondiagnostic_reference_qualified_evidence_qualifies(tmp_path):
    _run(tmp_path, qualified=("pickup_simple",))
    data = _payload(tmp_path)
    run = data["runs"][0]
    assert run["qualified"] == ["pickup_simple"]
    assert run["updates"] == 7 and run["consecutive_passes"]["pickup_simple"] == 3
    assert run["qualification_streaks"]["pickup_simple"]["evaluation_ids"] == ["eval-7", "eval-6", "eval-5"]
    assert len(run["latest_evaluations"]) == 2
    assert all(row["paired"] and row["reference_qualified"] and not row["diagnostic"]
               for row in run["latest_evaluations"])
    assert all(row["source_fingerprint"] and row["comparison_group"]
               and "evidence" not in row for row in run["latest_evaluations"])
    assert str(tmp_path) not in json.dumps(data)
    assert not list(tmp_path.rglob(".evidence-cache"))


def test_diagnostic_and_unqualified_reference_evidence_never_qualify(tmp_path):
    run = _run(tmp_path, qualified=("pickup_simple",), diagnostic=True)
    data = _payload(tmp_path)
    assert data["runs"][0]["qualified"] == []
    report_path = run / "evaluation-7.json"
    report = json.loads(report_path.read_text())
    report["diagnostic"] = False
    for category in report["evaluation_results"][0]["categories"]:
        category["reference_qualified"] = False
    report_path.write_text(json.dumps(report))
    assert _payload(tmp_path)["runs"][0]["qualified"] == []


def test_mismatched_or_malformed_evidence_is_excluded_and_reported(tmp_path):
    run = _run(tmp_path, qualified=("pickup_simple",))
    report_path = run / "evaluation-7.json"
    report = json.loads(report_path.read_text())
    report["sources"] = {"sim": "other-source"}
    report_path.write_text(json.dumps(report))
    data = _payload(tmp_path)
    assert data["runs"][0]["qualified"] == []
    assert data["errors"]
    report_path.write_text("{")
    assert _payload(tmp_path)["errors"]


def test_reported_state_cannot_bypass_scenario_gate_or_versions(tmp_path):
    run = _run(tmp_path, qualified=("pickup_simple",))
    report_path = run / "evaluation-7.json"
    report = json.loads(report_path.read_text())
    report["evaluation_results"][0]["scenario_version"] = "old"
    for category in report["evaluation_results"][0]["categories"]:
        category["successes"] = 10
    report_path.write_text(json.dumps(report))
    assert _payload(tmp_path)["runs"][0]["qualified"] == []


def test_nonprimitive_registry_gate_checks_success_and_throughput():
    scenario = SCENARIOS["active_collect_carry_score"]
    row = {"scenario": scenario.scenario_id, "category": scenario.required_categories[0],
           "scenario_version": scenario.version, "curriculum_version": CURRICULUM_VERSION,
           "split": "validation", "diagnostic": False, "paired": True,
           "reference_qualified": True, "episodes": 100, "successes": 95,
           "success_rate": .95, "useful_throughput": 1.0, "throughput_ratio": .81}
    assert _category_passes(row, scenario, scenario.required_categories[0], CURRICULUM_VERSION)
    row["throughput_ratio"] = .79
    assert not _category_passes(row, scenario, scenario.required_categories[0], CURRICULUM_VERSION)
    row["throughput_ratio"] = .81
    row["success_rate"] = .89
    assert not _category_passes(row, scenario, scenario.required_categories[0], CURRICULUM_VERSION)


def test_qualified_stage_requires_matching_checkpoint_and_protocol(tmp_path):
    run = _run(tmp_path, qualified=("pickup_simple",))
    stats_path = run / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["curriculum_state"]["last_qualified_checkpoint"]["pickup_simple"] = "other-policy"
    stats_path.write_text(json.dumps(stats))
    assert _payload(tmp_path)["runs"][0]["qualified"] == []


def test_duplicate_scenario_results_cannot_prove_qualification(tmp_path):
    run = _run(tmp_path, qualified=("pickup_simple",))
    report_path = run / "evaluation-7.json"
    report = json.loads(report_path.read_text())
    report["evaluation_results"].append(report["evaluation_results"][0])
    report_path.write_text(json.dumps(report))
    data = _payload(tmp_path)
    assert data["runs"][0]["qualified"] == []
    assert data["errors"]


def test_malformed_and_out_of_root_queue_lanes_are_isolated(tmp_path):
    queue = tmp_path / "orchestration.json"
    queue.write_text(json.dumps({"lanes": [None,
        {"directory": str(tmp_path.parent / "outside"), "status": "running"},
        {"directory": "runs/unknown", "architecture": "fixture"}]}))
    data = _payload(tmp_path)
    assert len(data["runs"]) == 1
    assert data["runs"][0]["status"] == "unknown"
    assert len(data["errors"]) == 2
    assert str(tmp_path) not in json.dumps(data)


def test_old_training_status_and_replay_routes_remain_available(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        assert json.loads(urlopen(base + "/api/training-status").read())["status"] == "waiting"
        assert isinstance(json.loads(urlopen(base + "/api/curriculum-status").read())["scenarios"], dict)
        replays = json.loads(urlopen(base + "/api/scenario-replays").read())
        assert "replays" in replays
    finally:
        server.shutdown()
        server.server_close()
