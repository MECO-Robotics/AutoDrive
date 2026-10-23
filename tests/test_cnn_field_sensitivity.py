"""Tensor intervention and record comparison only; no simulator/GPU runs."""
import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_cnn_field_sensitivity.py"
spec = importlib.util.spec_from_file_location("cnn_field_sensitivity", SCRIPT)
sensitivity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensitivity)


def test_mask_preserves_original_and_every_other_input():
    field = torch.arange(2*12*4*8, dtype=torch.float32).reshape(2, 12, 4, 8) + 1
    original = field.clone()
    inputs = {"field": field, "scalars": torch.randn(2, 137),
              "entities": torch.randn(2, 4, 12), "entity_mask": torch.ones(2, 4, dtype=torch.bool)}
    result = sensitivity.mask_perception_channels(inputs)
    assert result is not inputs
    torch.testing.assert_close(field, original, rtol=0, atol=0)
    for channel in range(12):
        if channel in sensitivity.MASKED_CHANNELS:
            assert torch.count_nonzero(result["field"][:, channel]) == 0
        else:
            torch.testing.assert_close(result["field"][:, channel], field[:, channel], rtol=0, atol=0)
    for name in ("scalars", "entities", "entity_mask"):
        assert result[name] is inputs[name]


def test_record_comparison_reports_differences_without_asserting_parity():
    actual = {key: [1., 2.] for key in sensitivity.PARITY_FIELDS}
    reference = {key: [1., 2.] for key in sensitivity.PARITY_FIELDS}
    assert sensitivity.compare_records(actual, reference)["exact_equal"]
    reference["action_counts"] = [1., 4.]
    reference["frames"] = [1.]
    result = sensitivity.compare_records(actual, reference)
    assert not result["exact_equal"]
    assert result["different_leaf_count"] == 2
    assert result["shape_mismatch_count"] == 1
    assert result["max_abs_numeric_difference"] == 2
    assert result["first_differing_paths"] == ["action_counts[1]", "frames.length"]


def test_decision_trace_uses_masked_categorical_actions():
    logits = torch.tensor([[0., 2., 100., 0., 0., 0., 0., 0.],
                           [0., 0., 100., 0., 0., 3., 0., 4.]])
    scalars = torch.zeros(2, 137)
    scalars[0, -7] = 1.  # Action 1 is the only eligible action.
    scalars[1, -3] = 1.  # Actions 5 and 7 are eligible.
    scalars[1, -1] = 1.
    selected = sensitivity.masked_categorical_actions(logits, scalars)
    assert selected.tolist() == [1, 7]
    assert logits.argmax(-1).tolist() == [2, 2]
    assert torch.bincount(selected, minlength=8).tolist() == [0, 1, 0, 0, 0, 0, 0, 1]
