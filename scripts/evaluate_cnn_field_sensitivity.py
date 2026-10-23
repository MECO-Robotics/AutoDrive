#!/usr/bin/env python3
"""Posthoc CNN raster missing-input sensitivity on actual perception simulations.

Dry-run is default. This is an out-of-distribution input intervention on frozen
weights, not a retrained ablation or proof of what caused training gains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
MASKED_CHANNELS = (2, 3, 4, 5, 8, 11)
SCENARIOS = ("pickup_simple", "pickup_scattered", "deterministic_defense", "defense_3v3",
             "offense_vs_deterministic_defense")
PARITY_FIELDS = ("acquired_per_world", "score_blue_per_world", "score_red_per_world",
                 "score_difference_per_world", "mean_acquired", "mean_score_difference",
                 "mean_red_score_integral", "red_score_integral_per_world",
                 "first_pickup_seconds_per_world", "first_red_score_seconds_per_world",
                 "timing_resolution_seconds", "seed", "ticks", "physics_dt",
                 "action_counts", "perception", "frames")


def runtime_hash():
    files = sorted((REPO / "frc_defense").glob("*.py"))
    files += sorted((REPO / "frc_defense/csrc").glob("*"))
    files += sorted((REPO / "frc_defense/dashboard_static").glob("*"))
    hashes = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files if path.is_file()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def mask_perception_channels(inputs):
    masked = dict(inputs)
    field = inputs["field"].clone()
    field[:, MASKED_CHANNELS, :, :] = 0
    masked["field"] = field
    return masked


def masked_categorical_actions(logits, scalars):
    from torch.distributions import Categorical
    mask = scalars[:, -8:] > .5
    return Categorical(logits=logits.masked_fill(~mask, -1e9)).logits.argmax(-1)


def compare_records(actual, reference, fields=PARITY_FIELDS):
    """Record exact differences without assuming cross-run bitwise parity."""
    differences, paths, shape_mismatches, numeric_max = 0, [], 0, 0.

    def visit(a, b, path):
        nonlocal differences, shape_mismatches, numeric_max
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys()):
                if key not in a or key not in b:
                    note(path + "." + key)
                else:
                    visit(a[key], b[key], path + "." + key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                shape_mismatches += 1
                note(path + ".length")
            for index, (left, right) in enumerate(zip(a, b)):
                visit(left, right, f"{path}[{index}]")
        elif a != b:
            note(path)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if math.isfinite(a) and math.isfinite(b):
                    numeric_max = max(numeric_max, abs(a-b))

    def note(path):
        nonlocal differences
        differences += 1
        if len(paths) < 20:
            paths.append(path)

    for key in fields:
        if key not in actual or key not in reference:
            note(key + ".missing")
        else:
            visit(actual[key], reference[key], key)
    return {"exact_equal": differences == 0, "different_leaf_count": differences,
            "shape_mismatch_count": shape_mismatches,
            "max_abs_numeric_difference": numeric_max, "first_differing_paths": paths,
            "compared_fields": list(fields)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-manifest", "--source-manifest", dest="manifest", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=REPO / "evaluations/architecture-study-20261020/matrix")
    parser.add_argument("--output", type=Path, default=REPO / "evaluations/architecture-study-20261020/field-sensitivity-v5")
    parser.add_argument("--run", action="store_true", help="execute two simulator evaluations; otherwise inspect only")
    args = parser.parse_args()
    expected = json.loads(args.manifest.read_text())["source_sha256"]
    if runtime_hash() != expected:
        parser.error("current runtime does not match --expected-manifest")
    case = args.matrix / "full_field_cnn" / args.scenario / f"seed-{args.seed}-lr-0.0003"
    metadata = json.loads((case / "metadata.json").read_text())
    original = json.loads((case / "eval-0004.json").read_text())
    checkpoint = case / "policy-0004.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if metadata.get("status") != "complete" or metadata["provenance"]["source_sha256"] != expected:
        parser.error("case is incomplete or uses another runtime source")
    if (metadata["architecture"] != "full_field_cnn" or metadata["scenario"] != args.scenario
            or metadata["seed"] != args.seed):
        parser.error("case metadata does not identify the requested CNN scenario/seed")
    if original["source_sha256"] != expected or original["checkpoint_sha256"] != checkpoint_hash:
        parser.error("saved final evaluation/checkpoint provenance mismatch")
    config = metadata["config"]
    if config["eval_envs"] != 4 or config["action_repeat"] != 10:
        parser.error("expected main-study four-world evaluation and repeat=10")
    destination = args.output / args.scenario / f"seed-{args.seed}"
    plan = {"mode": "run" if args.run else "dry-run", "architecture": "full_field_cnn",
            "scenario": args.scenario, "training_seed": args.seed,
            "evaluation_seed": args.seed + 100000, "device": args.device,
            "evaluation_worlds": 4, "ticks": config["eval_ticks"], "action_repeat": 10,
            "source_sha256": expected, "checkpoint_sha256": checkpoint_hash,
            "checkpoint": str(checkpoint.resolve()), "main_evaluation": str((case / "eval-0004.json").resolve()),
            "masked_raster_channels": list(MASKED_CHANNELS),
            "preserved": "raster geometry/self/team/hubs; all scalars and entity inputs; simulator perception",
            "diagnostic": "posthoc missing-input/OOD sensitivity; not retrained ablation or causal training evidence",
            "output": str(destination.resolve())}
    if not args.run:
        print(json.dumps(plan, indent=2))
        return 0
    if destination.exists() and any(destination.iterdir()):
        parser.error("output case already exists; use a fresh --output to retain prior diagnostics")

    sys.path.insert(0, str(REPO))
    import torch
    from frc_defense.architecture_benchmark import ARCHITECTURES, evaluate
    from frc_defense.architecture_scenarios import make_env

    torch.set_num_threads(1)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    if config.get("disable_miopen", False):
        torch.backends.cudnn.enabled = False
    model = ARCHITECTURES["full_field_cnn"]()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload["provenance"]["source_sha256"] != expected or payload["seed"] != args.seed
            or payload["scenario"] != args.scenario or payload["update"] != 4):
        parser.error("checkpoint payload does not identify the expected frozen final policy")
    model.load_state_dict(payload["model_state_dict"])
    model.to(args.device).eval()

    class RecordingPolicy(torch.nn.Module):
        def __init__(self, policy, mask_field):
            super().__init__()
            self.policy = policy
            self.mask_field = mask_field
            self.actions = []

        def forward(self, inputs):
            policy_inputs = mask_perception_channels(inputs) if self.mask_field else inputs
            logits, value = self.policy(policy_inputs)
            selected = masked_categorical_actions(logits, policy_inputs["scalars"])
            self.actions.append(selected.detach())
            return logits, value

    ids = (3, 4, 5) if args.scenario == "defense_3v3" else (3,)
    env = make_env(args.scenario, 4, args.device, args.seed + 100000,
                   horizon=config["eval_ticks"])
    normal_policy = RecordingPolicy(model, False)
    masked_policy = RecordingPolicy(model, True)
    normal = evaluate(normal_policy, env, args.scenario, args.seed + 100000, ids,
                      config["eval_ticks"], 10)
    ablated = evaluate(masked_policy, env, args.scenario, args.seed + 100000, ids,
                       config["eval_ticks"], 10)
    if runtime_hash() != expected:
        raise RuntimeError("runtime changed during sensitivity evaluation; records cannot be certified")
    for condition, record in (("normal", normal), ("masked_raster", ablated)):
        record.update(condition=condition, source_sha256=expected, checkpoint_sha256=checkpoint_hash,
                      training_seed=args.seed, architecture="full_field_cnn")
    for record, policy in ((normal, normal_policy), (ablated, masked_policy)):
        trace = torch.stack(policy.actions)
        trace_counts = torch.bincount(trace.flatten(), minlength=8).cpu().tolist()
        if trace_counts != record["action_counts"]:
            raise RuntimeError("Decision trace does not match evaluator masked categorical action counts")
        record["learned_robot_ids"] = list(ids)
        record["decision_ticks"] = list(range(0, config["eval_ticks"], 10))
        record["decision_actions"] = trace.reshape(-1, 4, len(ids)).cpu().tolist()
        record["decision_trace_action_counts"] = trace_counts
        record["decision_trace_matches_evaluator_counts"] = True
    trace_pairs = [(left, right) for normal_step, masked_step in zip(normal["decision_actions"], ablated["decision_actions"])
                   for normal_world, masked_world in zip(normal_step, masked_step)
                   for left, right in zip(normal_world, masked_world)]
    first_pairs = [(left, right) for normal_world, masked_world in zip(normal["decision_actions"][0], ablated["decision_actions"][0])
                   for left, right in zip(normal_world, masked_world)]
    parity = compare_records(normal, original)
    parity_metrics = compare_records(normal, original, [key for key in PARITY_FIELDS if key != "frames"])
    intervention = compare_records(ablated, normal)
    summary = {**plan, "status": "complete" if parity["exact_equal"] else "complete_with_normal_parity_warning",
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "normal_vs_main": parity, "normal_vs_main_metrics": parity_metrics,
               "masked_vs_normal": intervention,
               "decision_action_comparison": {
                   "changed": sum(left != right for left, right in trace_pairs), "total": len(trace_pairs),
                   "first_decision_changed": sum(left != right for left, right in first_pairs),
                   "first_decision_total": len(first_pairs),
                   "histogram_l1_difference": sum(abs(left-right) for left, right in zip(normal["action_counts"], ablated["action_counts"]))},
               "metric_changes_masked_minus_normal": {
                   key: ablated[key]-normal[key]
                   for key in ("mean_acquired", "mean_score_difference", "mean_red_score_integral")},
               "normal_metrics": {key: normal[key] for key in PARITY_FIELDS if key != "frames"},
               "masked_metrics": {key: ablated[key] for key in PARITY_FIELDS if key != "frames"},
               "interpretation": ("Changed observations can cause policy/trajectory differences. This posthoc intervention "
                                  "is out of distribution, retains scalar redundancy, and does not prove the raster caused learning gains.")}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "normal.json").write_text(json.dumps(normal, indent=2))
    (destination / "masked-raster.json").write_text(json.dumps(ablated, indent=2))
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": summary["status"], "scenario": args.scenario, "seed": args.seed,
                      "normal_matches_main": parity["exact_equal"],
                      "masked_metric_changes": summary["metric_changes_masked_minus_normal"],
                      "output": str(destination.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
