"""Compare A2/A3 model forward and backward against the pre-reuse block."""
import argparse
import ast
import copy
import json
import statistics
import subprocess
import time
import types

import torch

from frc_defense import meco_tracks
from frc_defense.meco_device_lease import DeviceLease
from frc_defense.meco_gpu_attestation import verify_post_lease_wx9100


def original_forward():
    source = subprocess.check_output([
        "git", "show", "f87acd21c:frc_defense/meco_tracks.py"], text=True)
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "SE2MessageBlock")
    func = next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == "forward")
    namespace = dict(vars(meco_tracks))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])),
                 "<original-se2-block>", "exec"), namespace)
    return namespace["forward"]


def inputs(nodes, device):
    features = torch.randn(1, nodes, 9, device=device)
    features[..., 6:8] = torch.rand(1, nodes, 2, device=device)
    features[..., 8] = torch.randint(8, (1, nodes), device=device).float()
    return dict(node_features=features,
        edges=torch.randint(nodes, (1, nodes, 10), device=device),
        edge_mask=torch.rand(1, nodes, 10, device=device) > .2,
        region_ids=(torch.arange(nodes, device=device) % 64)[None],
        globals_=torch.randn(1, 7, device=device),
        node_valid=torch.ones(1, nodes, dtype=torch.bool, device=device),
        coarse_edges=torch.arange(64, device=device).reshape(1, 64, 1),
        coarse_edge_mask=torch.ones(1, 64, 1, dtype=torch.bool, device=device))


def evaluate(model, data):
    model.zero_grad(set_to_none=True)
    result = model(**data)
    loss = (result["utility"].square().mean() + result["penalty"].mean() +
            result["behavior_logits"].square().mean())
    loss.backward()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nodes", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with DeviceLease([args.device], owner="plan-a-message-reuse-benchmark") as lease:
        attestation = verify_post_lease_wx9100(lease, (args.device,))
        torch.cuda.set_device(args.device)
        torch.manual_seed(681)
        torch.use_deterministic_algorithms(True)
        forward = original_forward()
        report = {"device": args.device, "nodes": args.nodes,
                  "repeats": args.repeats, "attestation": attestation, "tracks": {}}
        for track in ("A2", "A3"):
            candidate = meco_tracks.make_meco_track(track).to(args.device)
            baseline = copy.deepcopy(candidate)
            for module in baseline.modules():
                if isinstance(module, meco_tracks.SE2MessageBlock):
                    module.forward = types.MethodType(forward, module)
            data = inputs(args.nodes, args.device)
            old = evaluate(baseline, data)
            new = evaluate(candidate, data)
            torch.cuda.synchronize()
            differences = {key: float((old[key] - new[key]).abs().max())
                for key in ("utility", "penalty", "behavior_logits", "hidden")}
            differences["parameter_grad"] = max(
                float((a.grad - b.grad).abs().max())
                for a, b in zip(baseline.parameters(), candidate.parameters())
                if a.grad is not None and b.grad is not None)
            differences["status_equal"] = bool(torch.equal(
                old["device_input_valid"], new["device_input_valid"]))
            times = {"baseline_ms": [], "candidate_ms": []}
            for repeat in range(args.repeats):
                order = [("baseline_ms", baseline), ("candidate_ms", candidate)]
                if repeat % 2:
                    order.reverse()
                for name, model in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    evaluate(model, data)
                    torch.cuda.synchronize()
                    times[name].append((time.perf_counter() - start) * 1000)
            report["tracks"][track] = {**times, "differences": differences,
                "baseline_median_ms": statistics.median(times["baseline_ms"]),
                "candidate_median_ms": statistics.median(times["candidate_ms"])}
            print(json.dumps({track: report["tracks"][track]}), flush=True)
            del baseline, candidate, data, old, new
        print(json.dumps(report, default=str), flush=True)


if __name__ == "__main__":
    main()
