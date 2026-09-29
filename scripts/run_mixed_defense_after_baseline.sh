#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/brian/AutoDrive
baseline_run="$project_dir/checkpoints/rebuilt-defense-generational"
mixed_run="$project_dir/checkpoints/rebuilt-defense-mixed-gpu-adstar80"
baseline_checkpoint="$baseline_run/policy.pt"
python_bin="$project_dir/.venv/bin/python"
baseline_status="$baseline_run/status.json"
eval_seed=30000

while true; do
    state=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "missing"))' "$baseline_status")
    case "$state" in
        completed) break ;;
        failed) echo "Baseline training failed; mixed run will not start." >&2; exit 1 ;;
        running)
            if ! systemctl --user is-active --quiet frc-defense-train-generational.service; then
                echo "Baseline service stopped before reporting completion; mixed run will not start." >&2
                exit 1
            fi
            sleep 15
            ;;
        *) echo "Unexpected baseline status: $state" >&2; exit 1 ;;
    esac
done

test -s "$baseline_checkpoint"
"$python_bin" -m frc_defense.tensor_training evaluate \
    --task defense --opponent mixed --checkpoint "$baseline_checkpoint" \
    --episodes 64 --envs 64 --seed "$eval_seed" --horizon 750 --device cuda:1 \
    --output "$baseline_run/evaluation/mixed-heldout.json"

"$python_bin" -m frc_defense.tensor_training train \
    --task defense --opponent mixed --algorithm generational \
    --generations 20 --population 8 --elites 2 --envs 256 --seed 2028 \
    --initial-checkpoint "$baseline_checkpoint" \
    --output "$mixed_run" --device cuda:1

"$python_bin" -m frc_defense.tensor_training evaluate \
    --task defense --opponent mixed --checkpoint "$mixed_run/policy.pt" \
    --episodes 64 --envs 64 --seed "$eval_seed" --horizon 750 --device cuda:1 \
    --output "$mixed_run/evaluation/mixed-heldout.json"
