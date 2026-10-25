from frc_defense import meco_plan_c_label_audit as audit
from frc_defense.meco_counterfactual import PLAN_C_REPEAT_TARGET_SCHEMA


def test_audit_counts_sampled_repeated_label_coverage_deterministically(monkeypatch, tmp_path):
    rows = [
        {"plan_c_counterfactuals": [
            {"descriptor": {"action": 1}}, {"descriptor": {"action": 4}},
            {"descriptor": {"action": 1}}],
         "plan_c_pairwise_labels": [
             {"stable_preference": True, "repeat_differences": [2.0, 1.0]},
             {"stable_preference": False, "repeat_differences": [1.0, -1.0]},
             {"stable_preference": False, "repeat_differences": [0.0, 2.0]}]},
        {"plan_c_counterfactuals": [
            {"descriptor": {"action": 6}}, {"descriptor": {"action": 1}}],
         "plan_c_pairwise_labels": [
             {"stable_preference": True, "repeat_differences": [-1.0, -2.0]}]},
    ]
    manifest = {
        "split": "train", "plan_c_target_schema": PLAN_C_REPEAT_TARGET_SCHEMA,
        "scenario_id": "scenario", "category": "single",
        "source_identity_sha256": "a" * 64, "source_runtime_device_bound": True,
        "plan_c_repeat_protocol": {"repeat_protocol_id": "repeat-id"},
        "seed_branch_summaries": {"7": {"skipped_count": 3,
            "skipped_reason_counts": {"fewer_than_two_plan_c_descriptors": 3}}},
    }
    monkeypatch.setattr(audit, "source_identity", lambda: {"source": "test"})
    monkeypatch.setattr(audit, "_read_counterfactual_dataset",
        lambda *args, **kwargs: (rows, "d" * 64, "m" * 64, {}, "protocol-id",
                                 "p" * 64, manifest))
    args = {"qualification_protocol": tmp_path / "qual.json"}
    first = audit.audit_dataset(tmp_path / "data.jsonl", **args)
    second = audit.audit_dataset(tmp_path / "data.jsonl", **args)
    assert first == second
    assert first["coverage"] == {
        "prefix_count": 2, "feasible_prefix_count": 2, "skipped_branch_count": 3,
        "skipped_branch_counts_by_reason": {"fewer_than_two_plan_c_descriptors": 3},
        "candidate_count": 5, "candidate_count_per_prefix": {"minimum": 2, "maximum": 3},
        "pair_comparison_count": 4,
        "pair_label_counts": {"stable_preference": 2,
            "tied_or_zero_repeat_difference": 1, "mixed_sign": 1},
        "candidate_descriptor_action_counts": {"1": 3, "4": 1, "6": 1},
        "categories": {"single": 2},
    }
    assert first["audit_identity"]
    assert "does not establish data sufficiency" in " ".join(first["limits"])


def test_audit_rejects_heldout_or_nonrepeated_manifest(monkeypatch, tmp_path):
    manifest = {"split": "heldout", "plan_c_target_schema": "other"}
    monkeypatch.setattr(audit, "source_identity", lambda: {})
    monkeypatch.setattr(audit, "_read_counterfactual_dataset",
        lambda *args, **kwargs: ([], "d", "m", {}, "p", "q", manifest))
    import pytest
    with pytest.raises(ValueError, match="train-split"):
        audit.audit_dataset(tmp_path / "data", qualification_protocol=tmp_path / "qual")
