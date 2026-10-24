import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frc_defense import meco_task_scheduler as scheduler_module
from frc_defense.meco_supervisor import (_parser as supervisor_parser,
    _train_command)
from frc_defense.meco_phase4 import (
    PHASE4_FACTORS, phase4_factor_digest, phase4_train_categories,
    phase4_train_protocol_id, phase4_training_metadata,
    validate_phase4_training_assignment,
)
from frc_defense.meco_task_scheduler import MecoTaskScheduler
from frc_defense.meco_track_training import MecoTrainConfig, _parser as train_parser


def _phase4_scheduler(monkeypatch, *, scenario="offense_vs_scripted_pressure"):
    categories = phase4_train_categories(scenario)
    assert categories
    monkeypatch.setattr(scheduler_module, "protocol_cases", lambda *_args, **_kwargs:
        (SimpleNamespace(seeds=(101, 202)),))
    monkeypatch.setattr(scheduler_module, "eligible_scenarios",
        lambda _state: (scenario,))
    monkeypatch.setattr(scheduler_module, "task_mixture",
        lambda *_args, **_kwargs: {scenario: 1.})
    monkeypatch.setattr(scheduler_module, "phase4_train_categories",
        lambda _scenario: (categories[0],))
    return MecoTaskScheduler(17, exposure_chunks_per_scenario=1,
                             phase4_train_exposure=True)


def test_phase4_factors_balance_deterministically_over_train_seeds(monkeypatch):
    first = _phase4_scheduler(monkeypatch)
    second = _phase4_scheduler(monkeypatch)
    assignments = []
    for cycle in range(1, 11):
        a = first.schedule(cycle, target_update=cycle * 100)
        b = second.schedule(cycle, target_update=cycle * 100)
        assert a == b
        assert a["phase4_split"] == "train"
        assert a["phase4_factor"]["kind"] in PHASE4_FACTORS
        assert a["category"] in phase4_train_categories(a["scenario_id"])
        assert a["seed"] in (101, 202)
        assert validate_phase4_training_assignment(a) == a["phase4_factor"]
        first.complete(a, {})
        second.complete(b, {})
        assignments.append((a["phase4_factor"]["kind"], a["seed"]))
    assert set(assignments) == {(factor, seed)
        for factor in PHASE4_FACTORS for seed in (101, 202)}
    assert len(set(assignments)) == len(PHASE4_FACTORS) * 2


def test_phase4_assignment_and_manifest_bind_factor_and_protocol(monkeypatch):
    scheduler = _phase4_scheduler(monkeypatch)
    assignment = scheduler.schedule(1)
    metadata = phase4_training_metadata(assignment)
    assert metadata["split"] == "train"
    assert metadata["version"] == assignment["phase4_training_version"]
    assert metadata["factor"] == assignment["phase4_factor"]
    assert metadata["factor_sha256"] == phase4_factor_digest(metadata["factor"])
    assert metadata["assignment_id"] == assignment["phase4_training_assignment_id"]
    assert metadata["protocol_id"] == assignment["phase4_training_protocol_id"]

    altered = dict(assignment, phase4_factor_sha256="0" * 64)
    with pytest.raises(ValueError, match="factor digest"):
        phase4_training_metadata(altered)
    altered = dict(assignment, phase4_factor={"kind": "observation_delay"})
    with pytest.raises(ValueError, match="frozen registered contract"):
        phase4_training_metadata(altered)


def test_returned_assignments_cannot_mutate_global_factor_or_pending_state(monkeypatch):
    scheduler = _phase4_scheduler(monkeypatch)
    protocol_before = phase4_train_protocol_id()
    assignment = scheduler.schedule(1)
    pending_before = json.loads(json.dumps(scheduler.state["pending_assignment"]))
    factor_kind = assignment["phase4_factor"]["kind"]
    assert assignment["phase4_factor"] is not PHASE4_FACTORS[factor_kind]
    assert scheduler.state["pending_assignment"]["phase4_factor"] is not assignment["phase4_factor"]

    assignment["phase4_factor"].clear()
    assignment["phase4_factor"]["kind"] = "mutated"
    assert PHASE4_FACTORS[factor_kind]["kind"] == factor_kind
    assert scheduler.state["pending_assignment"] == pending_before
    assert phase4_train_protocol_id() == protocol_before

    pending_copy = scheduler.schedule(1)
    pending_copy["phase4_factor"].clear()
    assert scheduler.state["pending_assignment"] == pending_before
    assert phase4_train_protocol_id() == protocol_before


@pytest.mark.parametrize("split", ["validation", "test", "calibration", "diagnostic"])
def test_phase4_training_assignment_rejects_nontrain_split(monkeypatch, split):
    assignment = _phase4_scheduler(monkeypatch).schedule(1)
    assignment["phase4_split"] = split
    with pytest.raises(ValueError, match="restricted to the train split"):
        validate_phase4_training_assignment(assignment)


def test_scheduler_prerequisites_and_opt_in_are_bound_to_state(monkeypatch):
    opted_in = MecoTaskScheduler(17, exposure_chunks_per_scenario=1,
                                 phase4_train_exposure=True)
    initial = opted_in.schedule(1)
    assert initial["scenario_id"] == "pickup_simple"
    assert "phase4_factor" not in initial

    ordinary = MecoTaskScheduler(17, exposure_chunks_per_scenario=1)
    assignment = ordinary.schedule(1)
    assert "phase4_factor" not in assignment
    with pytest.raises(ValueError, match="configuration mismatch"):
        MecoTaskScheduler(17, 1, ordinary.state, phase4_train_exposure=True)

    prior_state = dict(ordinary.state)
    prior_state.pop("phase4_train_exposure")
    prior_state.pop("phase4_exposure_counts")
    compatible = MecoTaskScheduler(17, 1, prior_state)
    assert compatible.state["phase4_exposure_counts"] == {}

    phase4 = _phase4_scheduler(monkeypatch)
    pending = phase4.schedule(1)
    restored = MecoTaskScheduler(17, 1, phase4.state, phase4_train_exposure=True)
    assert restored.schedule(1) == pending


def test_supervisor_exposure_switch_is_explicit_opt_in():
    args = supervisor_parser().parse_args([
        "--devices", "cuda:0,cuda:1", "--evaluation-device", "cuda:1",
        "--phase4-train-exposure"])
    assert args.phase4_train_exposure is True


def test_train_command_carries_bound_factor_assignment_to_worker(monkeypatch):
    assignment = _phase4_scheduler(monkeypatch).schedule(1)
    config = MecoTrainConfig(track="A1", scenario_id=assignment["scenario_id"],
        category=assignment["category"], seed=assignment["seed"], device="cuda:0")
    command = _train_command(config=config, output=Path("/tmp/A1"),
        run_id="run", stop_file=Path("/tmp/STOP"),
        abort_file=Path("/tmp/ABORT"), invocation_id="invocation",
        resume=None, protocol_id="protocol", task_assignment=assignment,
        source_hash="source-hash",
        native_extension_attestation={"schema": 1, "identity": {},
            "identity_sha256": "attestation-hash"})
    payload = json.loads(command[command.index("--task-assignment-json") + 1])
    assert payload == assignment
    assert command[command.index("--expected-source-sha256") + 1] == "source-hash"
    assert json.loads(command[command.index(
        "--native-extension-attestation-json") + 1])["schema"] == 1
    args = train_parser().parse_args(command[command.index("train"):])
    assert json.loads(args.task_assignment_json)["phase4_training_assignment_id"] == (
        assignment["phase4_training_assignment_id"])


def test_train_factor_domain_is_registered_pressure_only():
    for scenario in ("offense_vs_scripted_pressure", "full_match_offense"):
        allowed = phase4_train_categories(scenario)
        assert allowed
        assert all(any(part.startswith(("level_1", "level_2", "level_3", "level_4", "level_5"))
                       for part in category.split("/")) for category in allowed)
    for scenario, category in (("pickup_simple", "single"),
                               ("offense_vs_scripted_pressure", "active/open/level_0")):
        assignment = {"scenario_id": scenario, "category": category,
            "seed": 1, "cycle": 1,
            "training_seed_pool_id": "pool",
            "phase4_training_version": 1,
            "phase4_factor": PHASE4_FACTORS["observation_delay"],
            "phase4_split": "train", "phase4_factor_sha256": "x",
            "phase4_training_assignment_id": "x", "phase4_training_protocol_id": "x"}
        with pytest.raises(ValueError, match="outside the registered train domain"):
            validate_phase4_training_assignment(assignment)
