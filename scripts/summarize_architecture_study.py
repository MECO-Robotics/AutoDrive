#!/usr/bin/env python3
"""Summarize completed simulator architecture cases without ranking convergence."""
from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
import shlex
import sys
from statistics import mean

SCENARIOS = ("pickup_simple", "pickup_scattered", "deterministic_defense", "defense_3v3",
             "offense_vs_deterministic_defense")
COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#332288")
MODEL_COLORS = dict(zip(sorted(("baseline_mlp", "residual_mlp", "deep_sets", "entity_attention",
                               "full_field_cnn", "full_field_resnet", "field_fno")), COLORS))


def number(value):
    return f"{value:.3f}" if isinstance(value, (float, int)) else "n/a"


def link(path):
    return f"[{path.name}](<{path.resolve()}>)"


def read_cases(root):
    cases, skipped = [], []
    for path in sorted(root.rglob("metadata.json")):
        if {"qualification", "integration", "outdated-unrestricted-pickup"}.intersection(path.relative_to(root).parts[:-1]):
            continue
        ancestors = [parent for parent in path.parents if parent == root or root in parent.parents]
        if any((parent / "OUTDATED.json").exists() for parent in ancestors):
            skipped.append(f"{path}: excluded by ancestor OUTDATED.json")
            continue
        try:
            metadata = json.loads(path.read_text())
            if metadata.get("status") != "complete":
                skipped.append(f"{path}: status={metadata.get('status', 'missing')}")
                continue
            directory = path.parent
            curve = json.loads((directory / "learning_curve.json").read_text())
            curve = sorted(curve, key=lambda row: row["transitions"])
            if not curve or curve[0].get("update") != 0:
                raise ValueError("missing initialized evaluation")
            if curve[-1].get("update") != metadata["config"]["updates"]:
                raise ValueError("missing final evaluation")
            required = ("transitions", "mean_acquired", "mean_score_difference", "update")
            for point in curve:
                for key in required:
                    if key not in point:
                        raise ValueError(f"curve point lacks {key}")
            references = {}
            for path_reference in sorted(directory.glob("reference-*.json")):
                record = json.loads(path_reference.read_text())
                references[path_reference.stem.removeprefix("reference-")] = {
                    key: record.get(key) for key in
                    ("mean_acquired", "mean_score_difference", "mean_red_score_integral", "seed", "ticks")
                }
            training = [json.loads(line) for line in (directory / "training.jsonl").read_text().splitlines()
                        if line.strip()] if (directory / "training.jsonl").exists() else []
            cases.append({"metadata": metadata, "path": path, "curve": curve,
                          "references": references, "training": training})
        except (ValueError, KeyError, OSError, TypeError) as error:
            skipped.append(f"{path}: {error}")
    return cases, skipped


def protocol(case):
    # Sweep selectors do not alter a single case's rollout/evaluation protocol.
    ignored = {"architectures", "scenarios", "seeds", "learning_rates", "output", "device"}
    return json.dumps({key: value for key, value in case["metadata"]["config"].items()
                       if key not in ignored}, sort_keys=True)


def svg_plot(cases, destination, learning_rate, metric):
    """Prefer publication plotting tools; retain a dependency-free SVG fallback."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        _svg_plot_fallback(cases, destination, learning_rate, metric)
        return "stdlib SVG fallback"
    from textwrap import fill

    labels = {"mean_acquired": "FUEL acquisitions (count)",
              "mean_score_difference": "Blue − red score (points)",
              "mean_red_score_integral": "Red score integral (point·seconds; lower is better)"}
    architectures = sorted({case["metadata"]["architecture"] for case in cases})
    colors = {name: MODEL_COLORS.get(name, COLORS[i % len(COLORS)]) for i, name in enumerate(architectures)}
    present = [scenario for scenario in SCENARIOS
               if any(case["metadata"]["scenario"] == scenario for case in cases)]
    columns = min(2, max(1, len(present)))
    rows = (len(present) + columns - 1) // columns or 1
    fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 4.2 * rows + 1.5), squeeze=False)
    fig.subplots_adjust(top=.79 if rows == 1 else .88, bottom=.24 if rows == 1 else .15,
                        wspace=.3, hspace=.55)
    fig.suptitle(f"Architecture short screen · learning rate {learning_rate:g}\n"
                 "Thin: training seeds; thick: mean at shared checkpoints", fontsize=14)
    for axis, scenario in zip(axes.flat, present):
        relevant = [case for case in cases if case["metadata"]["scenario"] == scenario
                    and all(metric in point for point in case["curve"])]
        axis.set_title(fill("3v3 opening probe" if scenario == "defense_3v3"
                            else scenario.replace("_", " "), 40), fontsize=11)
        axis.set_xlabel("Learned-agent decision transitions (count)")
        axis.set_ylabel(labels[metric])
        axis.grid(alpha=.2)
        axis.axhline(0, color="0.6", linewidth=.6)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        if not relevant:
            axis.text(.5, .5, "No completed cases with this metric", ha="center", va="center",
                      transform=axis.transAxes, color="0.4")
            continue
        by_model = defaultdict(list)
        for case in relevant:
            by_model[case["metadata"]["architecture"]].append(case)
        for model, model_cases in sorted(by_model.items()):
            for case in model_cases:
                axis.plot([p["transitions"] for p in case["curve"]],
                          [p[metric] for p in case["curve"]], color=colors[model],
                          alpha=.23, linewidth=1, marker=".", markersize=3)
            shared = set.intersection(*(set(p["transitions"] for p in case["curve"])
                                        for case in model_cases))
            xs = sorted(shared)
            ys = [mean(next(p[metric] for p in case["curve"] if p["transitions"] == x)
                       for case in model_cases) for x in xs]
            axis.plot(xs, ys, color=colors[model], linewidth=2.1, marker="o", markersize=3)
        maximum = max(p["transitions"] for case in relevant for p in case["curve"])
        axis.set_xlim(0, maximum or 1)
    for axis in list(axes.flat)[len(present):]:
        axis.remove()
    handles = [Line2D([0], [0], color=colors[model], linewidth=2, label=model)
               for model in architectures]
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015),
                   ncol=4 if columns == 2 else 2, frameon=False, fontsize=10)
    fig.savefig(destination, format="svg", metadata={"Title": labels[metric]})
    fig.savefig(destination.with_suffix(".png"), format="png", dpi=180)
    plt.close(fig)
    return f"matplotlib {matplotlib.__version__} (Agg; SVG and PNG)"


def _svg_plot_fallback(cases, destination, learning_rate, metric):
    """Use exact checkpoint coordinates; means only join checkpoints all seeds share."""
    width, height = 1200, 1150
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" role="img" aria-label="Architecture learning curves">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font:12px sans-serif;fill:#222}.title{font-size:16px;font-weight:bold}</style>',
             f'<text x="30" y="25" class="title">Short architecture screen: learning rate {learning_rate:g}</text>']
    architectures = sorted({case["metadata"]["architecture"] for case in cases})
    colors = {name: MODEL_COLORS.get(name, COLORS[i % len(COLORS)]) for i, name in enumerate(architectures)}
    for index, scenario in enumerate(SCENARIOS):
        left, top = 75 + (index % 2) * 590, 90 + (index // 2) * 330
        plot_width, plot_height = 485, 235
        relevant = [case for case in cases if case["metadata"]["scenario"] == scenario
                    and all(metric in point for point in case["curve"])]
        points = [point for case in relevant for point in case["curve"]]
        x_max = max((point["transitions"] for point in points), default=1) or 1
        values = [point[metric] for point in points]
        y_min, y_max = min([0] + values), max([0] + values)
        if y_min == y_max:
            y_max = y_min + 1
        margin = (y_max - y_min) * .06
        y_min, y_max = y_min - margin, y_max + margin
        def coords(x, y):
            return left + x / x_max * plot_width, top + plot_height * (y_max - y) / (y_max - y_min)
        parts.append(f'<text x="{left}" y="{top-15}" class="title">{scenario}</text>')
        label = {"mean_acquired": "FUEL acquisitions", "mean_score_difference": "Blue − red score",
                 "mean_red_score_integral": "Red cumulative score integral (score·seconds; lower is better)"}[metric]
        parts.append(f'<text x="{left}" y="{top-1}">{label}</text>')
        for tick in range(5):
            y = y_min + (y_max-y_min) * tick / 4
            _, sy = coords(0, y)
            parts.extend([f'<path d="M{left},{sy}h{plot_width}" stroke="#ddd"/>',
                          f'<text x="{left-7}" y="{sy+4}" text-anchor="end">{y:.2g}</text>'])
            x = x_max * tick / 4
            sx, _ = coords(x, 0)
            parts.append(f'<text x="{sx}" y="{top+plot_height+18}" text-anchor="middle">{x:.0f}</text>')
        parts.append(f'<text x="{left+plot_width/2}" y="{top+plot_height+36}" text-anchor="middle">Actual agent decision transitions</text>')
        by_model = defaultdict(list)
        for case in relevant:
            by_model[case["metadata"]["architecture"]].append(case)
        for model, model_cases in sorted(by_model.items()):
            for case in model_cases:
                xy = ' '.join(f'{x:.2f},{y:.2f}' for x, y in (coords(point["transitions"], point[metric]) for point in case["curve"]))
                parts.append(f'<polyline points="{xy}" fill="none" stroke="{colors[model]}" stroke-width="1" opacity="0.25"/>')
            shared = set.intersection(*(set(p["transitions"] for p in case["curve"]) for case in model_cases))
            averaged = []
            for transition in sorted(shared):
                ys = [next(p[metric] for p in case["curve"] if p["transitions"] == transition) for case in model_cases]
                averaged.append(coords(transition, mean(ys)))
            xy = ' '.join(f'{x:.2f},{y:.2f}' for x, y in averaged)
            parts.append(f'<polyline points="{xy}" fill="none" stroke="{colors[model]}" stroke-width="2.5"/>')
        if not relevant:
            parts.append(f'<text x="{left+10}" y="{top+40}">No completed cases</text>')
    for i, model in enumerate(architectures):
        x, y = 35 + (i % 4)*290, 1095 + (i//4)*23
        parts.extend([f'<path d="M{x},{y}h20" stroke="{colors[model]}" stroke-width="3"/>',
                      f'<text x="{x+27}" y="{y+4}">{escape(model)}</text>'])
    parts.append('</svg>')
    destination.write_text('\n'.join(parts))


def summarize(root, output):
    cases, skipped = read_cases(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    protocols = sorted({protocol(case) for case in cases})
    ids = {settings: f"P{i+1}" for i, settings in enumerate(protocols)}
    lines = ["# Neural architecture simulator screen", "",
             "This is a short learning screen, not evidence of convergence or statistical superiority. "
             "Independent training seeds, not evaluation worlds, determine replication. Three seeds remain weak evidence. "
             "Means and paired changes below are descriptive; no significance claims are made.", "",
             "This report is a snapshot. Do not treat the study as final until all 105 planned main-matrix cases "
             "and the declared learning-rate sweep are complete and reviewed.", "",
             f"Source root: `{root.resolve()}`. Completed cases: **{len(cases)}**; excluded/incomplete metadata: **{len(skipped)}**.", "",
             "See [research and evaluation rationale](NN_ARCHITECTURE_RESEARCH.md). "
             "Policy inputs use simulated perception; truth may be used to calculate evaluation outcomes. "
             "This summary reads recorded metrics, not replay frames, and does not verify dashboard freshness.", "",
             "Acquisition is the recorded cumulative FUEL acquisition count, which can count reacquisition. "
             "Score difference is blue minus red. Δ columns subtract each seed's initialized evaluation before averaging. "
             "Evaluation uses greedy masked actions; training samples categorical actions. "
             "Training seconds exclude evaluation. GPU slots 0 and 1 are the same AMD WX 9100 model; device slot is ignored for sample-learning grouping "
             "and preserved per case. Wall times are descriptive and can reflect hardware placement or overlapping jobs; they are not isolated speed benchmarks. "
             "Other protocol differences remain separate.", ""]
    lines.extend(["The experimental offense_vs_deterministic_defense scenario places blue neural offense against a deterministic red defender. "
                  "Its research factory uses generic strategic execution and lacks deterministic-only intake/scoring refinements; "
                  "results describe this limited execution path, not a production offense replacement.", "",
                  "On HIP, disable_miopen defaults true to avoid the residual CNN wrw_reduction_hip weight-gradient fault. "
                  "The same ATen convolution backend is used across alternatives. "
                  f"Backend qualification: {link(root.parent / 'qualification' / 'neural-backend-disabled.json')}. "
                  "That record covers three field encoders at batches 32 and 96; all seven learning-matrix alternatives use the selected backend. "
                  "Qualification and integration artifacts are excluded from the learning matrix.", ""])
    lines.extend(["Pickup scenarios mask scoring, interception, and ferry actions 4/5/6. "
                  "This prevents the previously observed collect/release/reacquire loop. "
                  "Earlier unrestricted-pickup trials were moved to outdated-unrestricted-pickup and marked OUTDATED.json; "
                  "they are excluded from this matrix. No architecture conclusion follows from those trials.", ""])
    groups = defaultdict(list)
    for case in cases:
        metadata = case["metadata"]
        groups[(metadata["scenario"], metadata["learning_rate"], ids[protocol(case)], metadata["architecture"])].append(case)
    for scenario in SCENARIOS:
        lines.extend([f"## {scenario}", ""])
        relevant = [(key, group) for key, group in sorted(groups.items()) if key[0] == scenario]
        if not relevant:
            lines.extend(["No completed cases.", ""])
            continue
        lines.extend(["| LR | Protocol | Model | Cases / seeds | Params | Transitions | Acq initial → final | Δ acq | Score initial → final | Δ score | Train seconds |",
                      "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for key, group in relevant:
            _, lr, config_id, architecture = key
            metadata = [case["metadata"] for case in group]
            starts, ends = [c["curve"][0] for c in group], [c["curve"][-1] for c in group]
            def average(rows, name):
                return mean(row[name] for row in rows)
            params = sorted({m["parameters"] for m in metadata})
            transitions = sorted({p["transitions"] for p in ends})
            lines.append(f"| {lr:g} | {config_id} | {architecture} | {len(group)} / {len({m['seed'] for m in metadata})} | "
                         f"{', '.join(map(str,params))} | {', '.join(map(str,transitions))} | "
                         f"{number(average(starts,'mean_acquired'))} → {number(average(ends,'mean_acquired'))} | "
                         f"{number(mean(b['mean_acquired']-a['mean_acquired'] for a,b in zip(starts,ends)))} | "
                         f"{number(average(starts,'mean_score_difference'))} → {number(average(ends,'mean_score_difference'))} | "
                         f"{number(mean(b['mean_score_difference']-a['mean_score_difference'] for a,b in zip(starts,ends)))} | "
                         f"{number(mean(m.get('train_seconds',0) for m in metadata))} |")
        lines.append("")
        scenario_cases = [c for c in cases if c["metadata"]["scenario"] == scenario]
        if not any(p["mean_score_difference"] != 0 for c in scenario_cases for p in c["curve"]):
            lines.extend(["**No nonzero net-score observations:** score differences do not distinguish these checkpoints. "
                          "Zero net score can include equal scoring by both teams; inspect per-team scores in records. "
                          "For defense this short screen cannot establish learned score suppression.", ""])
        if any(c["training"] and all(abs(row.get("mean_reward",0)) < 1e-12 for row in c["training"]) for c in scenario_cases):
            lines.extend(["**Sparse-reward warning:** at least one run logged zero mean training reward at every update. "
                          "A zero mean can also hide canceling rewards. Such runs do not establish successful acquisition of defense skill.", ""])
        for case in scenario_cases:
            records = case["training"]
            if records and all(all(key in row for key in ("acquisitions", "score_blue", "score_red")) for row in records):
                if not any(row["acquisitions"] or row["score_blue"] or row["score_red"] for row in records):
                    m = case["metadata"]
                    lines.extend([f"**No recorded acquisition or scoring events during training:** {m['architecture']}, "
                                  f"seed {m['seed']}, LR {m['learning_rate']:g}. Shaping reward alone does not establish task learning.", ""])
        lines.extend(["| LR | Protocol | Model | Red score integral initial → final | Δ integral | Final first pickup: reached / worlds; mean seconds among reached | Final first red score: reached / worlds; mean seconds among reached |",
                      "|---|---|---|---:|---:|---|---|"])
        for key, group in relevant:
            _, lr, config_id, architecture = key
            valid = [c for c in group if all("mean_red_score_integral" in p for p in (c["curve"][0], c["curve"][-1]))]
            if valid:
                initial = mean(c["curve"][0]["mean_red_score_integral"] for c in valid)
                final = mean(c["curve"][-1]["mean_red_score_integral"] for c in valid)
                integrals, delta = f"{number(initial)} → {number(final)}", number(final-initial)
            else:
                integrals, delta = "n/a", "n/a"
            def latency_summary(name):
                values = [v for c in group for v in c["curve"][-1].get(name, [])]
                reached = [v for v in values if v is not None]
                return f"{len(reached)} / {len(values)}; {number(mean(reached) if reached else None)}"
            lines.append(f"| {lr:g} | {config_id} | {architecture} | {integrals} | {delta} | "
                         f"{latency_summary('first_pickup_seconds_per_world')} | {latency_summary('first_red_score_seconds_per_world')} |")
        lines.extend(["", "Red score integral is a delay-sensitive secondary outcome: with the same terminal red score, "
                      "a smaller integral indicates later scoring within this horizon. It is sampled at the recorded action-repeat resolution, "
                      "not exact event time. A negative Δ integral is favorable for suppressing red offense. "
                      "Unreached first events are censored and excluded from the conditional latency mean; the reached/worlds denominator must be read with that mean. "
                      "Red defender scoring is not the central objective in the experimental blue-offense scenario.", ""])
    lines.extend(["## Learning curves", "", "Thin translucent lines are individual training seeds; thick lines are means at checkpoints shared by all available seeds for that model. "
                  "Curves remain separate by learning rate and protocol. Partial seed availability is not a balanced comparison. "
                  "Transitions count learned-agent decisions; 3v3 has three learned agents per world. Compare physical time using metadata and world_physics_ticks.", ""])
    for settings, config_id in ids.items():
        for lr in sorted({c["metadata"]["learning_rate"] for c in cases if protocol(c) == settings}):
            selected = [c for c in cases if protocol(c) == settings and c["metadata"]["learning_rate"] == lr]
            scenarios = {case["metadata"]["scenario"] for case in selected}
            metrics = []
            if any(scenario.startswith("pickup") or scenario == "offense_vs_deterministic_defense" for scenario in scenarios):
                metrics.append(("mean_acquired", "acquisitions"))
            if any(not scenario.startswith("pickup") for scenario in scenarios):
                metrics.append(("mean_score_difference", "score"))
            if scenarios.intersection({"deterministic_defense", "defense_3v3"}):
                metrics.append(("mean_red_score_integral", "red-score-integral"))
            for metric, label in metrics:
                chart = output.with_name(f"{output.stem}-{config_id}-lr-{lr:g}-{label}.svg")
                renderer = svg_plot(selected, chart, lr, metric)
                lines.extend([f"### {config_id}, learning rate {lr:g}, {label}", "", f"![Learning curves](<{chart.resolve()}>)", "",
                              f"Renderer: {renderer}." + (f" {link(chart.with_suffix('.png'))}" if renderer.startswith("matplotlib") else ""), ""])
    lines.extend(["## Per-seed records", "", "Every row links the exact case metadata, learning curve, and final evaluation. Reference columns show hold/scripted acquisition and net score; absent references are n/a. "
                  "Hold is an active hold tactic, not necessarily an absent physical defender. A true undefended-attacker baseline must be supplied separately before attributing prevented score.", "",
                  "| Scenario / model / LR / protocol / seed | Device | Acq initial → final (Δ) | Score initial → final (Δ) | Hold acq / score | Scripted acq / score | Train seconds | Records |",
                  "|---|---|---:|---:|---:|---:|---:|---|"])
    for case in cases:
        m, directory = case["metadata"], case["path"].parent
        a, b = case["curve"][0], case["curve"][-1]
        def reference(name):
            r = case["references"].get(name, {})
            return f"{number(r.get('mean_acquired'))} / {number(r.get('mean_score_difference'))}"
        record_links = [link(case["path"]), link(directory/"learning_curve.json"), link(directory/f"eval-{b['update']:04d}.json")]
        for file in sorted(directory.glob("reference-*.json")):
            record_links.append(link(file))
        if (directory/"training.jsonl").exists():
            record_links.append(link(directory/"training.jsonl"))
        lines.append(f"| {m['scenario']} / {m['architecture']} / {m['learning_rate']:g} / {ids[protocol(case)]} / {m['seed']} | "
                     f"{m['config'].get('device', 'unspecified')} | "
                     f"{number(a['mean_acquired'])} → {number(b['mean_acquired'])} ({number(b['mean_acquired']-a['mean_acquired'])}) | "
                     f"{number(a['mean_score_difference'])} → {number(b['mean_score_difference'])} ({number(b['mean_score_difference']-a['mean_score_difference'])}) | "
                     f"{reference('hold')} | {reference('scripted')} | {number(m.get('train_seconds'))} | {' · '.join(record_links)} |")
    lines.extend(["", "### Raw timing and training events per seed", "",
                  "Per-world arrays preserve null for unreached events. Timing resolution is in seconds. "
                  "Training counts sum the logged rollout events, not evaluation outcomes.", "",
                  "| Case metadata | Red score integral per world, initial → final | First pickup seconds, initial → final | First red score seconds, initial → final | Timing resolution | Training acquisitions / blue score / red score |",
                  "|---|---|---|---|---:|---:|"])
    for case in cases:
        a, b = case["curve"][0], case["curve"][-1]
        def raw_pair(name):
            return f"`{json.dumps(a.get(name))}` → `{json.dumps(b.get(name))}`"
        training = case["training"]
        totals = [sum(row[key] for row in training) if training and all(key in row for row in training) else None
                  for key in ("acquisitions", "score_blue", "score_red")]
        lines.append(f"| {link(case['path'])} | {raw_pair('red_score_integral_per_world')} | "
                     f"{raw_pair('first_pickup_seconds_per_world')} | {raw_pair('first_red_score_seconds_per_world')} | "
                     f"{number(b.get('timing_resolution_seconds'))} | {' / '.join(number(v) for v in totals)} |")
    lines.extend(["", "## Reproduction and settings", "",
                  "Reproduce this report:", "", "```bash",
                  f"{shlex.quote(sys.executable)} scripts/summarize_architecture_study.py --root {shlex.quote(str(root.resolve()))} --output {shlex.quote(str(output.resolve()))}", "```", "",
                  "Case metadata preserves the exact full CLI configuration, reward expression, seed, model size, and source hashes. "
                  "The following are protocol settings excluding matrix selectors, output location, and device slot. "
                  "Fixed PPO implementation details remain in the source identified by each case's provenance.", ""])
    for settings, config_id in ids.items():
        lines.extend([f"### {config_id}", "", "```json", json.dumps(json.loads(settings), indent=2, sort_keys=True), "```", ""])
        settings_dict = json.loads(settings)
        example = next(c for c in cases if protocol(c) == settings)
        metadata = example["metadata"]
        command = [".venv/bin/python", "-m", "frc_defense.architecture_benchmark",
                   "--architectures", metadata["architecture"], "--scenarios", metadata["scenario"],
                   "--seeds", str(metadata["seed"]), "--learning-rates", str(metadata["learning_rate"]),
                   "--output", str(root.resolve())]
        if "device" in metadata["config"]:
            command.extend(["--device", metadata["config"]["device"]])
        for name, value in settings_dict.items():
            option = "--"+name.replace("_", "-")
            if isinstance(value, bool):
                command.append(option if value else "--no-"+name.replace("_", "-"))
            else:
                command.extend([option, str(value)])
        lines.extend(["Example single-case reproduction (choose a fresh output path to preserve existing artifacts):", "", "```bash", shlex.join(command), "```", ""])
    sources = sorted({json.dumps(c["metadata"].get("provenance",{}), sort_keys=True) for c in cases})
    lines.extend(["### Source provenance", "", "```json", json.dumps([json.loads(s) for s in sources], indent=2), "```", ""])
    if len(sources) > 1:
        lines.extend(["**Multiple source provenances are present.** Review case metadata before attributing differences solely to architecture.", ""])
    if skipped:
        lines.extend(["## Excluded or incomplete cases", ""])
        lines.extend(f"- `{reason}`" for reason in skipped)
        lines.append("")
    output.write_text('\n'.join(lines))
    return {"completed": len(cases), "excluded": len(skipped), "output": str(output.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("evaluations/architecture-study-20261020/matrix"))
    parser.add_argument("--output", type=Path, default=Path("docs/NN_ARCHITECTURE_RESULTS.md"))
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("--root must be an existing directory")
    print(json.dumps(summarize(args.root, args.output)))


if __name__ == "__main__":
    main()
