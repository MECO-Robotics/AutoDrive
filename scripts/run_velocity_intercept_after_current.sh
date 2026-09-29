#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/brian/AutoDrive
python_bin="$project_dir/.venv/bin/python"
current_run="$project_dir/checkpoints/rebuilt-defense-mixed-gpu-adstar80"
next_run="$project_dir/checkpoints/rebuilt-defense-mixed-gpu-velocity-intercept"
current_status="$current_run/status.json"
current_supervisor_pid=661668

while true; do
    state=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "missing"))' "$current_status")
    case "$state" in
        completed) break ;;
        failed) echo "Current mixed run failed; velocity-intercept run will not start." >&2; exit 1 ;;
        running) sleep 20 ;;
        *) echo "Unexpected current-run status: $state" >&2; exit 1 ;;
    esac
done

# Let the current run's held-out evaluation finish before taking the GPU.
while kill -0 "$current_supervisor_pid" 2>/dev/null; do sleep 20; done

test -s "$current_run/policy.pt"
mkdir -p "$next_run"
"$python_bin" -c 'import json,sys; json.dump({"status":"queued","algorithm":"generational","opponent":"mixed","opponents":["adstar","offense","intercept","velocity_intercept","mirror"],"started_from":sys.argv[1]},open(sys.argv[2],"w"),indent=2)' \
    "$current_run/policy.pt" "$next_run/queued.json"

"$python_bin" -m frc_defense.tensor_training train \
    --task defense --opponent mixed --algorithm generational \
    --generations 20 --population 8 --elites 2 --envs 256 --seed 2029 \
    --initial-checkpoint "$current_run/policy.pt" \
    --output "$next_run" --device cuda:1

"$python_bin" -m frc_defense.tensor_training evaluate \
    --task defense --opponent mixed --checkpoint "$next_run/policy.pt" \
    --episodes 64 --envs 64 --seed 30000 --horizon 750 --device cuda:1 \
    --output "$next_run/evaluation/mixed-heldout.json"
