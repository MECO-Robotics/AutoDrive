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
initial_shard_id=$(printf 'd0-calibration-pickup_simple-shard-%04d-of-0008' "$initial_index")
initial_manifest="$manifest_dir/$initial_shard_id.json"

initial_worker_is_running() {
  local proc_cmdline="/proc/$initial_pid/cmdline"
  local -a proc_args=()
  local has_device=0 has_manifest=0 has_command=0 has_shard=0 i

  [[ -d "/proc/$initial_pid" ]] || return 1
  if [[ ! -r "$proc_cmdline" ]] || ! mapfile -d '' -t proc_args < "$proc_cmdline"; then
    if [[ ! -d "/proc/$initial_pid" ]]; then return 1; fi
    echo "cannot read live PID $initial_pid command line; refusing to continue" >&2
    return 2
  fi

  for ((i = 0; i + 1 < ${#proc_args[@]}; i++)); do
    [[ "${proc_args[i]}" == "--device" && "${proc_args[i + 1]}" == "$device" ]] && has_device=1
    [[ "${proc_args[i]}" == "--manifest" && "${proc_args[i + 1]}" == "$initial_manifest" ]] && has_manifest=1
  done
  for i in "${proc_args[@]}"; do
    [[ "$i" == "calibrate-d0-shard" ]] && has_command=1
    [[ "$i" == *"shard-$(printf '%04d' "$initial_index")-of-0008.json" ]] || continue
    has_shard=1
  done

  if (( has_device && has_manifest && has_command && has_shard )); then return 0; fi
  echo "live PID $initial_pid does not match shard $initial_index on $device with manifest $initial_manifest; refusing to continue" >&2
  return 2
}

echo "$(date -u +%FT%TZ) waiting for shard $initial_index on $device (pid $initial_pid)"
while :; do
  if initial_worker_is_running; then
    sleep 5
  else
    worker_status=$?
    [[ "$worker_status" -eq 1 ]] && break
    exit "$worker_status"
  fi
done

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
