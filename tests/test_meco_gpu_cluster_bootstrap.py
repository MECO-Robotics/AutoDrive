"""Deferred HIP fixtures for the GPU v4 bootstrap tranche.

These cases must run only after the campaign releases the WX9100 devices and
the separately approved GPU verification window is open.
"""
from types import SimpleNamespace

import pytest


def _hip_device_or_skip(torch):
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("requires the leased HIP device verification window")
    return torch.device("cuda", torch.cuda.current_device())


def test_device_cluster_bootstrap_rejects_cpu_inputs():
    import torch
    from frc_defense.meco_qualification import _device_cluster_bootstrap_ci

    with pytest.raises(ValueError, match="explicitly indexed HIP"):
        _device_cluster_bootstrap_ci(
            {("layout", "family"): torch.tensor([0.0])},
            seed=1, device="cpu", replicates=8)


def test_device_cluster_bootstrap_degenerate_and_unequal_clusters():
    import torch
    from frc_defense.meco_qualification import _device_cluster_bootstrap_ci

    device = _hip_device_or_skip(torch)
    constant = {
        ("layout-a", "family-a"): torch.tensor([2.5, 2.5], device=device),
        ("layout-b", "family-b"): torch.tensor([2.5], device=device),
    }
    lower, upper = _device_cluster_bootstrap_ci(
        constant, seed=17, device=device, replicates=128, replicate_chunk=16)
    assert lower.device == device and upper.device == device
    assert lower.item() == upper.item() == 2.5

    unequal = {
        ("layout-a", "family-a"): torch.tensor([0.0, 2.0], device=device),
        ("layout-b", "family-b"): torch.tensor([10.0], device=device),
        ("layout-c", "family-c"): torch.tensor([4.0, 6.0, 8.0], device=device),
    }
    first = _device_cluster_bootstrap_ci(
        unequal, seed=991, device=device, replicates=256, replicate_chunk=32)
    second = _device_cluster_bootstrap_ci(
        unequal, seed=991, device=device, replicates=256, replicate_chunk=32)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert 0.0 <= first[0].item() <= first[1].item() <= 10.0


def test_device_category_statistics_binds_paired_gate_verdict():
    import torch
    from frc_defense.meco_qualification import (
        SAFETY_METRIC_NAMES, _device_category_statistics)
    from frc_defense.training_curriculum import GateDefinition

    device = _hip_device_or_skip(torch)
    cases = [SimpleNamespace(layout_id="layout-a", defender_family="family-a"),
             SimpleNamespace(layout_id="layout-b", defender_family="family-b")]

    def row(throughput, success, safety):
        metrics = {"useful_throughput": throughput, "success": success}
        metrics.update({name: safety for name in SAFETY_METRIC_NAMES})
        return {"metrics": metrics}

    result = _device_category_statistics("pickup_simple", cases,
        [row(2.0, True, 0.1), row(3.0, True, 0.1)],
        [row(1.0, True, 0.1), row(2.0, True, 0.1)],
        [row(1.0, True, 0.1), row(2.0, True, 0.1)],
        device=device, evaluation_id="fixture-eval", protocol_id="fixture-protocol",
        gate=GateDefinition(min_episodes=1, success_rate=1.0), useful_margin=0.0,
        safety_margins={name: 0.0 for name in SAFETY_METRIC_NAMES}, seed=33,
        reference_policy="D1")
    assert result["category_result"]["episodes"] == 2
    assert result["category_result"]["successes"] == 2
    assert result["category_result"]["success_rate"] == 1.0
    assert result["category_verdict"]["success_rate_pass"] is True
    assert result["category_verdict"]["useful_gain_ci_pass"] is True
    assert result["category_verdict"]["safety_pass"] is True

    zero_reference = _device_category_statistics("pickup_simple", cases,
        [row(1.0, True, 0.1), row(2.0, True, 0.1)],
        [row(0.0, True, 0.1), row(0.0, True, 0.1)],
        [row(1.0, True, 0.1), row(2.0, True, 0.1)],
        device=device, evaluation_id="zero-reference", protocol_id="fixture-protocol",
        gate=GateDefinition(min_episodes=1, success_rate=1.0,
                            throughput_ratio=0.5), useful_margin=0.0,
        safety_margins={name: 1.0 for name in SAFETY_METRIC_NAMES}, seed=34,
        reference_policy="D1")
    assert zero_reference["category_verdict"]["throughput_ratio_pass"] is False


def test_retention_verdict_requires_explicit_scope_and_safety_disposition():
    from frc_defense.training_curriculum import (
        CURRICULUM_VERSION, CategoryResult, EvaluationResult, _failures,
        get_scenario, seed_partition)

    scenario = get_scenario("pickup_simple")
    seeds = seed_partition(scenario.scenario_id, "validation",
                           scenario.gate.min_episodes, master_seed=51)
    categories = tuple(CategoryResult(name, len(seeds), len(seeds),
        1.0, 1.0, True, True, "fixture-protocol", seeds, seeds)
        for name in scenario.required_categories)
    device = {name: {"success_rate_pass": True,
        "throughput_ratio_pass": True, "minimum_useful_throughput_pass": True,
        "useful_gain_ci_pass": True, "safety_pass": None,
        "safety_assessment": "not_in_retention_gate", "safety_metric_pass": {},
        "retention_baseline_pass": True}
        for name in scenario.required_categories}
    verdict = {"schema_version": 1,
        "verdict_type": "meco-gpu-numeric-gate-verdict-v1",
        "implementation_id": "meco-v4-cluster-episode-device-percentile-v1",
        "verdict_scope": "retention", "evaluation_id": "fixture-eval",
        "protocol_id": "fixture-protocol",
        "runtime_device": {"backend": "hip", "device": "cuda:0"},
        "category_verdicts": device, "failures": []}
    result = EvaluationResult("fixture-eval", scenario.scenario_id, "validation",
        categories, "fixture-bundle", scenario.version, CURRICULUM_VERSION,
        "fixture-protocol", verdict,
        protocol_version="meco-qualification-v5-gpu-bootstrap")
    assert _failures(result, expected_verdict_scope="retention") == []
    assert "GPU numeric gate verdict is missing or unbound" in _failures(result)

    malformed = {**verdict, "verdict_scope": "qualification"}
    malformed_result = EvaluationResult("fixture-eval", scenario.scenario_id,
        "validation", categories, "fixture-bundle", scenario.version,
        CURRICULUM_VERSION, "fixture-protocol", malformed,
        protocol_version="meco-qualification-v5-gpu-bootstrap")
    assert any("GPU numeric category verdict is invalid" in failure
               for failure in _failures(malformed_result))

    failing_categories = tuple(CategoryResult(name, len(seeds), 0,
        0.0, 0.0, True, True, "fixture-protocol", seeds, seeds)
        for name in scenario.required_categories)
    missing_v4 = EvaluationResult("missing-verdict", scenario.scenario_id,
        "validation", failing_categories, "fixture-bundle", scenario.version,
        CURRICULUM_VERSION, "fixture-protocol",
        protocol_version="meco-qualification-v5-gpu-bootstrap")
    missing_failures = _failures(missing_v4)
    assert "GPU numeric gate verdict is missing or unbound" in missing_failures
    assert not any("success gate failed" in failure or
                   "absolute delivery minimum failed" in failure
                   for failure in missing_failures)

    legacy = EvaluationResult("legacy", scenario.scenario_id, "validation",
        failing_categories, "fixture-bundle", scenario.version,
        CURRICULUM_VERSION, "legacy-protocol")
    assert any("success gate failed" in failure for failure in _failures(legacy))
