"""Study orchestration contracts; no Torch imports or simulated behavior."""
import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_architecture_study_v5.py"
spec = importlib.util.spec_from_file_location("architecture_study_v5", SCRIPT)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_plan_has_119_unique_cases_and_exact_budgets(tmp_path):
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    queues = study.build_plan(tmp_path / "matrix", python)
    assert [len(queues[slot]) for slot in (0, 1)] == [51, 68]
    cases = [case for queue in queues.values() for case in queue]
    assert len({case["directory"] for case in cases}) == 119
    assert sum(case["learning_rate"] == .0003 for case in cases) == 105
    assert all(case["scenario"] == "deterministic_defense" and case["seed"] == 7
               for case in cases if case["learning_rate"] != .0003)
    for case in cases:
        command = case["command"]
        assert command[0] == str(python.absolute())  # Preserve virtualenv symlink path.
        horizon, rollout = study.SCENARIO_BUDGETS[case["scenario"]]
        for flag, value in (("--horizon", horizon), ("--eval-ticks", horizon),
                            ("--rollout-steps", rollout), ("--updates", 4),
                            ("--envs", 8), ("--eval-envs", 4),
                            ("--eval-every", 2), ("--action-repeat", 10)):
            assert command[command.index(flag) + 1] == str(value)
        assert "--disable-miopen" in command


def test_two_worker_manifest_with_fresh_non_simulator_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "runtime_hash", lambda: "frozen-test-hash")
    output = tmp_path / "matrix"
    queues = {}
    for slot in (0, 1):
        directory = output / f"case-{slot}"
        code = ("import json,sys;from pathlib import Path;"
                "p=Path(sys.argv[1]);p.mkdir(parents=True);"
                "(p/'metadata.json').write_text(json.dumps({'status':'complete',"
                "'provenance':{'source_sha256':'frozen-test-hash'}}))")
        queues[slot] = [{"architecture": "fake", "scenario": "test", "seed": slot,
                         "learning_rate": .0003, "directory": str(directory),
                         "command": [sys.executable, "-c", code, str(directory)],
                         "status": "pending"}]
    assert study.run_plan(queues, output, "frozen-test-hash") == 0
    manifest = json.loads((output / "orchestration-v5.json").read_text())
    assert manifest["status"] == "complete" and manifest["completed_cases"] == 2
    assert all(queue[0]["status"] == "complete" for queue in queues.values())
    assert len({queue[0]["pid"] for queue in queues.values()}) == 2
