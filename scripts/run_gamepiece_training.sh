#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 TASK DEVICE SEED OUTPUT_DIR" >&2
    exit 2
fi

task=$1
device=$2
seed=$3
output=$4
project_dir=/home/brian/AutoDrive
python_bin="$project_dir/.venv/bin/python"
checkpoint="$output/policy.pt"

args=("$python_bin" -m frc_defense.tensor_training train
    --task "$task" --algorithm generational --architecture strategic_adstar
    --generations 40 --opponent-pool-size 8 --strong-checkpoints 2 --envs 24 --horizon 8000
    --seed "$seed" --output "$output" --device "$device")

if [[ -s "$checkpoint" ]]; then
    args+=(--initial-checkpoint "$checkpoint")
fi

exec "${args[@]}"
