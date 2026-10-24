#!/usr/bin/env bash
# Run at most one scheduler per device. Do not launch a separate shard job on
# that device while this scheduler is active: flock below coordinates scheduler
# instances only, not independently launched calibration workers.
set -euo pipefail

repo_root="/mnt/zfs-storage/All Drives/Brian Data/Projects/AutoDrive"
device="$1"
initial_pid="$2"
initial_index="$3"
shift 3

manifest_device="${device//:/-}"
output_dir="$repo_root/evaluations/meco-five/d0-calibration-20261024/source-31e5/$manifest_device"
manifest_dir="$repo_root/qualifications/meco/shards/pickup-simple-20261024/source-31e5/$manifest_device"

initial_worker_is_running() {
  local proc_cmd
  [[ -r "/proc/$initial_pid/cmdline" ]] || return 1
  proc_cmd=$(tr '\0' ' ' < "/proc/$initial_pid/cmdline")
  [[ "$proc_cmd" == *"calibrate-d0-shard"* &&
     "$proc_cmd" == *"shard-$(printf '%04d' "$initial_index")-of-0008.json"* ]]
}

echo "$(date -u +%FT%TZ) waiting for shard $initial_index on $device (pid $initial_pid)"
while initial_worker_is_running; do sleep 5; done

mkdir -p "$output_dir"
exec 9>"$output_dir/.remaining-shards.lock"
flock -n 9 || { echo "another shard scheduler already owns $device" >&2; exit 1; }

for shard_index in "$initial_index" "$@"; do
  shard_id=$(printf 'd0-calibration-pickup_simple-shard-%04d-of-0008' "$shard_index")
  shard_file=$(printf 'shard-%04d.artifact.json' "$shard_index")
  manifest="$manifest_dir/$shard_id.json"
  output="$output_dir/$shard_file"
  log="$output_dir/shard-$(printf '%04d' "$shard_index").log"
  mkdir -p "$output_dir"
  echo "$(date -u +%FT%TZ) starting shard $shard_index on $device"
  cd "$repo_root"
  "$repo_root/.venv/bin/python" -c 'import sys; from frc_defense.fuel_physics_hip import prepare_native_fuel_extension; prepare_native_fuel_extension(sys.argv[1]); from frc_defense.meco_qualification import main; main(sys.argv[2:])' \
    "$device" calibrate-d0-shard --manifest "$manifest" --device "$device" \
    --output "$output" >> "$log" 2>&1
  echo "$(date -u +%FT%TZ) completed shard $shard_index on $device"
done
