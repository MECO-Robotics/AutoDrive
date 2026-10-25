"""Dashboard curriculum payload contract; saved fixtures only, no simulation."""
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

from frc_defense.dashboard import create_handler, _merge_meco_track_status
from frc_defense.dashboard_api import curriculum_status_payload, _category_passes, _curriculum_report_root
from frc_defense.training_curriculum import CURRICULUM_VERSION, SCENARIOS
from scripts.report_curriculum_progress import summarize


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


def test_live_meco_assignment_overrides_run_config_on_active_tracks():
    payload = {"tracks": [{"track": track} for track in ("A1", "A2", "A3", "C1", "C2")]}
    policies = [{"policy_id": track, "training_state": "running",
        "evaluation_state": "not_evaluated", "run": {"completed_updates": 4,
            "config": {"scenario_id": "config_scenario"}, "device": "cuda:0"}}
        for track in ("A1", "A2", "A3", "C1", "C2")]
    lanes = {track: {"status": "running"} for track in ("A1", "A2", "A3", "C1", "C2")}
    status = {"policies": policies, "queue": {"state": "fresh", "lanes": lanes, "curriculum": {
        "state": "fresh", "assignment": {"scenario_id": "live_scenario",
            "category": "live_category"}}}}

    merged = json.loads(_merge_meco_track_status(json.dumps(payload).encode(), status))

    assert all(track["current"] == "live_scenario" for track in merged["tracks"])
    assert all(track["current_category"] == "live_category" for track in merged["tracks"])
    assert all(track["current_source"] == "live_assignment" for track in merged["tracks"])


def test_deferred_lane_does_not_inherit_sibling_live_assignment():
    payload = {"tracks": [{"track": track} for track in ("A1", "A2")]}
    policies = [
        {"policy_id": "A1", "training_state": "unconfirmed",
         "run": {"config": {"scenario_id": "configured-a1"}}},
        {"policy_id": "A2", "training_state": "running",
         "run": {"config": {"scenario_id": "configured-a2"}}},
    ]
    status = {"policies": policies, "queue": {"state": "fresh", "lanes": {
        "A1": {"status": "deferred_for_evaluation"},
        "A2": {"status": "running"}}, "curriculum": {
            "state": "fresh", "assignment": {"scenario_id": "live-scenario",
                "category": "live-category"}}}}

    merged = json.loads(_merge_meco_track_status(json.dumps(payload).encode(), status))

    assert merged["tracks"][0]["current"] == "configured-a1"
    assert merged["tracks"][0]["current_source"] == "run_config_fallback"
    assert merged["tracks"][1]["current"] == "live-scenario"
    assert merged["tracks"][1]["current_category"] == "live-category"


def test_stale_queue_suppresses_live_assignment_even_if_payload_has_one():
    payload = {"tracks": [{"track": "A1"}]}
    status = {"policies": [{"policy_id": "A1", "training_state": "running",
        "run": {"config": {"scenario_id": "configured"}}}],
        "queue": {"state": "stale", "lanes": {"A1": {"status": "running"}},
            "curriculum": {"state": "fresh", "assignment": {
                "scenario_id": "stale-scenario", "category": "old"}}}}

    merged = json.loads(_merge_meco_track_status(json.dumps(payload).encode(), status))

    assert merged["tracks"][0]["current"] == "configured"
    assert merged["tracks"][0]["current_source"] == "run_config_fallback"


def test_run_config_current_is_labeled_fallback_when_assignment_is_unavailable():
    payload = {"tracks": [{"track": "A1"}]}
    status = {"policies": [{"policy_id": "A1", "training_state": "idle",
        "run": {"config": {"scenario_id": "config_scenario"}}}],
        "queue": {"lanes": {}, "curriculum": {"state": "stale", "assignment": None}}}

    merged = json.loads(_merge_meco_track_status(json.dumps(payload).encode(), status))

    assert merged["tracks"][0]["current"] == "config_scenario"
    assert merged["tracks"][0]["current_category"] is None
    assert merged["tracks"][0]["current_source"] == "run_config_fallback"


def test_curriculum_api_merge_preserves_saved_evaluation_evidence(tmp_path, monkeypatch):
    saved = {"tracks": [{"track": "A1", "status": "saved-status",
        "evaluation": "saved-evaluation", "runs": [{"run_id": "evidence-run"}]}],
        "scenarios": {"pickup_simple": {"qualified": True}}}
    live = {"policies": [{"policy_id": "A1", "training_state": "running",
        "run": {"config": {"scenario_id": "configured"}}}],
        "queue": {"state": "fresh", "lanes": {"A1": {"status": "running"}},
            "curriculum": {"state": "fresh", "assignment": {
                "scenario_id": "live-scenario", "category": "live-category"}}}}
    monkeypatch.setattr("frc_defense.dashboard.curriculum_status_payload",
        lambda *args, **kwargs: json.dumps(saved).encode())
    monkeypatch.setattr("frc_defense.dashboard.meco_status_payload",
        lambda *args, **kwargs: json.dumps(live).encode())

    response = _http_payload(tmp_path)

    assert response["tracks"][0]["status"] == "running"
    assert response["tracks"][0]["current"] == "live-scenario"
    assert response["tracks"][0]["runs"] == [{"run_id": "evidence-run"}]
    assert response["scenarios"] == saved["scenarios"]


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


def test_curriculum_errors_use_bounded_queue_or_evidence_projection(tmp_path, monkeypatch):
    (tmp_path / "orchestration.json").write_text("{}")
    monkeypatch.setattr("scripts.report_curriculum_progress.summarize", lambda root: {
        "runs": [], "queue_lanes": [], "queue": None,
        "errors": [
            {"path": "/private/secret/orchestration.json", "error": "queue lane 2: TOKEN"},
            {"path": "/private/secret/evaluation-raw.json", "error": "TOKEN"},
        ],
    })

    payload = json.loads(curriculum_status_payload(
        tmp_path, curriculum_root=tmp_path, project_root=tmp_path))

    assert payload["errors"] == [
        {"error": "Training queue status could not be projected."},
        {"error": "Saved evaluation evidence could not be verified."},
    ]
    assert "private" not in json.dumps(payload)
    assert "TOKEN" not in json.dumps(payload)


def test_dashboard_requires_explicit_curriculum_root_when_run_dir_is_not_one(tmp_path):
    defense_root = tmp_path / "checkpoints" / "rebuilt-gamepiece-defense"
    defense_root.mkdir(parents=True)
    curriculum_root = tmp_path / "evaluations" / "curriculum-campaign"
    _run(curriculum_root)
    assert _curriculum_report_root(defense_root, tmp_path) is None
    assert _curriculum_report_root(defense_root, tmp_path, curriculum_root) == curriculum_root


def test_queued_only_lane_is_pending_and_stale_queue_is_reported(tmp_path):
    lane = {"track": "A1", "architecture": "fixture", "directory": "A1", "status": "queued",
            "current": "pickup_simple", "completed_updates": 0,
            "qualified": ["forged"], "evaluation": {"success": True}}
    (tmp_path / "A1").mkdir()
    queue = tmp_path / "queue-status.json"
    queue.write_text(json.dumps({"lanes": [lane]}))
    summary = summarize(tmp_path)
    assert summary["queue_lanes"] == [{"track": "A1", "status": "queued",
                                       "completed_updates": 0, "device": None,
                                       "current": "pickup_simple"}]
    assert summary["runs"][0]["evaluation_history"] == []
    data = _payload(tmp_path)
    assert len(data["runs"]) == 1
    assert data["runs"][0]["status"] == "queued"
    assert data["runs"][0]["current"] == "pickup_simple"
    assert data["runs"][0]["qualified"] == []
    assert data["runs"][0]["latest_evaluations"] == []
    assert "qualification_evidence" not in data["runs"][0]
    import os, time
    os.utime(queue, (time.time() - 200, time.time() - 200))
    assert _payload(tmp_path)["runs"][0]["status"] == "stale"


def test_current_five_track_run_directory_queue_resolves_checkpoint_root(tmp_path):
    tracks = ("A1", "A2", "A3", "C1", "C2")
    lanes = []
    for track in tracks:
        run = tmp_path / "checkpoints" / track
        run.mkdir(parents=True)
        (run / "stats.json").write_text(json.dumps({
            "curriculum_state": {}, "stats": [], "config": {"eval_every": 1}}))
        lanes.append({"track": track, "run_directory": track,
                      "status": "queued", "completed_updates": 0})
    (tmp_path / "queue-status.json").write_text(json.dumps({"lanes": lanes}))

    report = summarize(tmp_path)

    assert report["errors"] == []
    assert {run["run"] for run in report["runs"]} == {
        f"checkpoints/{track}" for track in tracks}
    assert {lane["track"] for lane in report["queue_lanes"]} == set(tracks)
    assert {Path(run["path"]).resolve() for run in report["runs"]} == {
        (tmp_path / "checkpoints" / track).resolve() for track in tracks}


def test_queue_directory_conflicts_and_unsafe_paths_fail_closed(tmp_path):
    for name in ("A1", "other"):
        (tmp_path / "checkpoints" / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / "linked").symlink_to(tmp_path / "other", target_is_directory=True)
    (tmp_path / "checkpoints" / "A3").symlink_to(
        tmp_path / "checkpoints" / "other", target_is_directory=True)
    (tmp_path / "checkpoints" / "A2").write_text("not a directory")
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    queue = tmp_path / "queue-status.json"

    invalid_lanes = [
        {"track": "A1", "run_directory": "A1", "directory": "checkpoints/other"},
        {"track": "A1", "run_directory": "../other"},
        {"track": "A1", "run_directory": "A1", "directory": "linked"},
        {"track": "A1", "run_directory": "A1", "directory": str(outside)},
        {"track": "missing", "run_directory": "missing"},
        {"track": "A1", "run_directory": "A1", "directory": "checkpoints/A1/child"},
        {"track": "A1", "run_directory": "A1", "directory": "checkpoints-link/A1"},
        {"track": "A2", "run_directory": "A2"},
        {"track": "A3", "run_directory": "A3"},
        {"track": "A1/child", "run_directory": "A1/child"},
    ]
    for lane in invalid_lanes:
        queue.write_text(json.dumps({"lanes": [lane]}))
        report = summarize(tmp_path)
        assert report["errors"]
        assert report["queue_lanes"] == []


def test_matching_legacy_and_current_queue_directories_are_unambiguous(tmp_path):
    run = tmp_path / "A1"
    run.mkdir()
    (run / "stats.json").write_text(json.dumps({
        "curriculum_state": {}, "stats": [], "config": {"eval_every": 1}}))
    (tmp_path / "queue-status.json").write_text(json.dumps({"lanes": [{
        "track": "A1", "run_directory": "A1", "directory": str(run.resolve()),
        "status": "queued", "completed_updates": 0}]}))

    report = summarize(tmp_path)

    assert report["errors"] == []
    assert [lane["track"] for lane in report["queue_lanes"]] == ["A1"]


def test_legacy_track_root_and_absolute_directory_remain_supported(tmp_path):
    run = tmp_path / "A1"
    run.mkdir()
    (run / "stats.json").write_text(json.dumps({
        "curriculum_state": {}, "stats": [], "config": {"eval_every": 1}}))
    (tmp_path / "queue-status.json").write_text(json.dumps({"lanes": [{
        "track": "A1", "directory": str(run.resolve()),
        "status": "running", "completed_updates": 3}]}))

    report = summarize(tmp_path)

    assert report["errors"] == []
    assert len(report["runs"]) == 1
    assert report["runs"][0]["path"] == str(run.resolve())
    assert report["queue_lanes"][0]["track"] == "A1"


def test_two_stats_backed_track_layouts_are_ambiguous_without_disambiguator(tmp_path):
    for run in (tmp_path / "checkpoints" / "A1", tmp_path / "A1"):
        run.mkdir(parents=True)
        (run / "stats.json").write_text(json.dumps({
            "curriculum_state": {}, "stats": [], "config": {"eval_every": 1}}))
    (tmp_path / "queue-status.json").write_text(json.dumps({"lanes": [{
        "track": "A1", "run_directory": "A1", "status": "running"}]}))

    report = summarize(tmp_path)

    assert any("ambiguous" in error["error"] for error in report["errors"])
    assert report["queue_lanes"] == []
    assert len(report["runs"]) == 2


def test_directory_disambiguates_two_stats_backed_layouts_only_by_exact_candidate(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "A1"
    legacy = tmp_path / "A1"
    for run in (checkpoint, legacy):
        run.mkdir(parents=True)
        (run / "stats.json").write_text(json.dumps({
            "curriculum_state": {}, "stats": [], "config": {"eval_every": 1}}))
    (tmp_path / "queue-status.json").write_text(json.dumps({"lanes": [{
        "track": "A1", "run_directory": "A1", "directory": str(legacy.resolve()),
        "status": "running"}]}))

    report = summarize(tmp_path)

    assert report["errors"] == []
    assert report["queue_lanes"][0]["track"] == "A1"
    selected = next(run for run in report["runs"] if run.get("queue_status") == "running")
    assert selected["path"] == str(legacy.resolve())


def test_same_basename_elsewhere_under_root_never_matches_queue_track(tmp_path):
    expected = tmp_path / "checkpoints" / "A1"
    unrelated = tmp_path / "other" / "A1"
    unrelated.mkdir(parents=True)
    (unrelated / "stats.json").write_text(json.dumps({
        "curriculum_state": {"qualified": ["forged"]}, "stats": [],
        "config": {"eval_every": 1}}))
    expected.mkdir(parents=True)
    (tmp_path / "queue-status.json").write_text(json.dumps({"lanes": [{
        "track": "A1", "run_directory": "A1", "status": "queued",
        "qualified": ["forged"]}]}))

    report = summarize(tmp_path)

    queued = next(run for run in report["runs"] if run["run"] == "checkpoints/A1")
    assert queued["qualified"] == []
    assert queued["latest_evaluations"] == []
    assert queued["evaluation_history"] == []
    assert queued["qualification_evidence"] == []


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
    (tmp_path / "A1").mkdir()
    queue = tmp_path / "orchestration.json"
    queue.write_text(json.dumps({"lanes": [None,
        {"directory": str(tmp_path.parent / "outside"), "status": "running"},
        {"track": "A1", "directory": "A1", "architecture": "fixture"}]}))
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
