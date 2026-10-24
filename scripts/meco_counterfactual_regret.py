"""Offline sampled-set regret for replay-bound Plan C counterfactual data.

The reported quantity is bounded-set regret over the exact sampled C rows in
counterfactual-v2. It is not regret against the full simulator action space or
against a globally optimal simulator policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frc_defense.curriculum_training import source_identity
from frc_defense.meco_supervised import (_materialize_counterfactual_rows,
    _read_counterfactual_dataset)
from frc_defense.meco_tracks import (MECO_CANDIDATE_RAW_FEATURE_COUNT, PREDICTION_MODES,
    make_meco_track, prediction_contract, score_candidates)


REPORT_SCHEMA = "meco-sampled-counterfactual-regret-v1"
REGRET_SEMANTICS = (
    "max gross_horizon_return among exact replay-bound sampled Plan C candidate "
    "outcomes minus gross_horizon_return of the checkpoint-selected candidate, "
    "computed only when the exact selected candidate ID has a recorded outcome")
SCORING_SEMANTICS = (
    "checkpoint C1/C2 candidate utility minus replayed route and prediction-risk "
    "costs; invalid table rows masked; argmax chooses the first row in stable "
    "candidate-table order on exact ties; recurrent hidden state is reconstructed "
    "by scoring every observation in that branch's verified teacher prefix")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode())


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _config_digest(config: dict) -> str:
    # The training update count can advance during a run; all other settings
    # define the checkpoint's inference configuration.
    immutable = {key: value for key, value in config.items() if key != "updates"}
    return _canonical_sha256(immutable)


def _load_c_checkpoint(track: str, checkpoint: Path, *, manifest: dict,
                       expected_sources: dict):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if (track not in ("C1", "C2") or not isinstance(config, dict) or
            config.get("track") != track or payload.get("sources") != expected_sources or
            payload.get("prediction_contract") != prediction_contract(config.get("prediction_mode")) or
            config.get("candidate_k") != manifest.get("candidate_k") or
            config.get("action_ticks") != manifest.get("max_macro_ticks") or
            config.get("prediction_mode") != manifest.get("prediction_mode") or
            config.get("scenario_id") != manifest.get("scenario_id") or
            config.get("category") != manifest.get("category") or
            config.get("device") != manifest.get("device", {}).get("device") or
            config.get("route_cost_points_per_second") != manifest.get("route_cost_points_per_second") or
            payload.get("qualification_protocol_sha256") != manifest.get("qualification_protocol_sha256") or
            payload.get("qualification_protocol_id") != manifest.get("protocol_id") or
            not isinstance(payload.get("model"), dict)):
        raise ValueError("checkpoint track, source, protocol, scenario, or scoring configuration differs from dataset")
    if config.get("prediction_mode") not in PREDICTION_MODES:
        raise ValueError("checkpoint has an unsupported prediction mode")
    model = make_meco_track(track,
        candidate_features=MECO_CANDIDATE_RAW_FEATURE_COUNT)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("checkpoint weights do not match the requested C scorer") from exc
    model.eval()
    return model, payload, config


def _replay_teacher_prefix_and_score(row: dict, manifest: dict, sample: dict,
                                     model, *, track: str, device: str):
    """Rebuild the exact recurrent context and score the archived branch table."""
    from frc_defense.curriculum_scenarios import make_curriculum_env
    from frc_defense.meco_counterfactual import (_counterfactual_rng_scope,
        _hash, _prefix_snapshot, _reward_step, _verify_native_fuel_extension)
    from frc_defense.meco_track_training import (_reset_episode,
        _snapshot_tensors, _tracking_reset_hidden)
    from frc_defense.meco_pose_graph import build_legal_pose_graph

    scenario_id, category, seed = row["scenario_id"], row["category"], row["seed"]
    prefix_transcript = row.get("prefix_transcript")
    if (not isinstance(prefix_transcript, list) or
            len(prefix_transcript) != row.get("decision_index") or
            _hash(prefix_transcript) != row.get("prefix_transcript_sha256")):
        raise ValueError("teacher prefix is missing or its digest/length is invalid")
    runtime_device = manifest["device"]["device"]
    hidden = None
    tracking_was_available = None
    replay_prefix = []
    with _counterfactual_rng_scope(seed):
        env = make_curriculum_env(scenario_id, category, 1, runtime_device,
            seed, horizon=manifest["horizon_physics_ticks"])
        try:
            _verify_native_fuel_extension(manifest["native_extension_identity"])
            graph = build_legal_pose_graph(env, slot=3)
            episode_id = f"counterfactual-{scenario_id}-{category}-{seed}"
            _reset_episode(env, seed)
            elapsed_ticks = 0
            for index, expected in enumerate(prefix_transcript):
                ticks = expected.get("macro_ticks")
                remaining = manifest["horizon_physics_ticks"] - elapsed_ticks
                if (type(ticks) is not int or ticks != min(
                        manifest["max_macro_ticks"], remaining)):
                    raise ValueError("teacher prefix has invalid macro cadence")
                current = _prefix_snapshot(env, graph, episode_id,
                    action_ticks=ticks, candidate_k=manifest["candidate_k"],
                    prediction_mode=manifest["prediction_mode"],
                    max_c_candidates=manifest["max_plan_c_descriptors_per_prefix"],
                    max_a_pairs=manifest["max_plan_a_pairs_per_prefix"],
                    materialize=True)
                mapping = {
                    "snapshot_id": "snapshot_id",
                    "predecision_simulator_state_sha256": "simulator_state_sha256",
                    "predecision_observation_sha256": "observation_sha256",
                    "safe_observation_sha256": "safe_observation_sha256",
                    "candidate_digest": "candidate_digest",
                    "candidate_replay_digest": "candidate_replay_digest",
                    "support_digest": "support_digest",
                    "macro_ticks": "macro_ticks",
                }
                for archived, actual in mapping.items():
                    if expected.get(archived) != current.get(actual):
                        raise RuntimeError(
                            f"teacher prefix replay mismatch at decision {index}: {archived}")
                action = expected.get("action")
                if (type(action) is not int or action != current["teacher_action"] or
                        expected.get("branch_eligibility") != current["branch_eligibility"]):
                    raise RuntimeError("teacher action or eligibility differs from exact replay")
                inputs = current["_training_inputs"]
                raster, globals_ = _snapshot_tensors(inputs["observation"],
                    device=device)
                hidden_before, tracking_available, _ = _tracking_reset_hidden(
                    hidden, tracking_was_available, raster)
                candidate_set = inputs["candidate_set"]
                with torch.inference_mode():
                    decision = score_candidates(track, model, raster=raster,
                        globals_=globals_, table=candidate_set.table,
                        hidden=hidden_before,
                        route_cost=inputs["route_cost"].to(device),
                        risk_cost=inputs["risk_cost"].to(device))
                hidden = decision["hidden"]
                tracking_was_available = tracking_available
                reward, ended, timeout, trace, _, _, _ = _reward_step(env, action,
                    ticks, goal_xy=expected.get("goal_xy_m"),
                    goal_heading=expected.get("goal_heading_rad"),
                    allow_action7_traversal=expected.get("allow_action7_traversal", False))
                if (trace != expected.get("executed_action_trace") or
                        reward != expected.get("gross_macro_return")):
                    raise RuntimeError("teacher prefix reward or action trace differs from replay")
                if ended or timeout:
                    raise RuntimeError("teacher prefix terminates before archived branch point")
                replay_prefix.append({**current,
                    "prefix_goal_xy_m": expected.get("goal_xy_m"),
                    "prefix_goal_heading_rad": expected.get("goal_heading_rad"),
                    "prefix_action7_traversal": expected.get("allow_action7_traversal", False),
                    "executed_action_trace": trace,
                    "prefix_gross_return": reward,
                    "prefix_candidate_commitment": None})
                elapsed_ticks += len(trace)

            ticks = row.get("macro_ticks")
            if type(ticks) is not int or ticks != min(
                    manifest["max_macro_ticks"],
                    manifest["horizon_physics_ticks"] - elapsed_ticks):
                raise ValueError("branch macro cadence differs from replay horizon")
            branch = _prefix_snapshot(env, graph, episode_id,
                action_ticks=ticks, candidate_k=manifest["candidate_k"],
                prediction_mode=manifest["prediction_mode"],
                max_c_candidates=manifest["max_plan_c_descriptors_per_prefix"],
                max_a_pairs=manifest["max_plan_a_pairs_per_prefix"],
                materialize=True)
            archived_branch = row.get("branch_state")
            if not isinstance(archived_branch, dict):
                raise ValueError("branch state provenance is missing")
            if (branch["branch_eligibility"] != row.get("branch_eligibility") or
                    not branch["branch_eligibility"].get("eligible")):
                raise RuntimeError("branch eligibility differs from strict replay")
            for field in ("snapshot_id", "simulator_state_sha256", "observation_sha256",
                    "safe_observation_sha256", "candidate_digest",
                    "candidate_replay_digest", "support_digest"):
                if archived_branch.get(field) != branch[field]:
                    raise RuntimeError(f"branch snapshot differs from verified replay at {field}")
            replayed_set = branch["_training_inputs"]["candidate_set"]
            if (replayed_set.table.candidate_ids != sample["table"].candidate_ids or
                    replayed_set.table.valid_mask != sample["table"].valid_mask or
                    branch["candidate_replay_digest"] != row["candidate_replay_digest"]):
                raise RuntimeError("scored candidate table differs from strict materialized replay")
            raster, globals_ = _snapshot_tensors(
                branch["_training_inputs"]["observation"], device=device)
            hidden_before, _, _ = _tracking_reset_hidden(hidden,
                tracking_was_available, raster)
            with torch.inference_mode():
                decision = score_candidates(track, model, raster=raster,
                    globals_=globals_, table=replayed_set.table,
                    hidden=hidden_before,
                    route_cost=branch["_training_inputs"]["route_cost"].to(device),
                    risk_cost=branch["_training_inputs"]["risk_cost"].to(device))
            branch_teacher_action = row.get("branch_teacher_action_reference")
            if (type(branch_teacher_action) is not int or
                    branch_teacher_action != branch["teacher_action"]):
                raise RuntimeError("branch teacher action differs from replay")
            reward, ended, timeout, trace, _, _, _ = _reward_step(env,
                branch_teacher_action, ticks)
            replay_prefix.append({**branch,
                "prefix_goal_xy_m": None,
                "prefix_goal_heading_rad": None,
                "prefix_action7_traversal": False,
                "executed_action_trace": trace,
                "prefix_gross_return": reward,
                "prefix_candidate_commitment": None})
            return decision["logits"].reshape(-1), replay_prefix
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()


def _verify_sampled_branch_outcomes(row: dict, manifest: dict, replay_prefix: list):
    """Re-execute every sampled descriptor and fail on any return drift."""
    from frc_defense.meco_counterfactual import _branch_return

    expected_fields = ("gross_macro_return", "gross_horizon_return",
        "discounted_horizon_return", "continuation_policy",
        "continuation_macro_horizon", "discount_semantics", "terminal")
    outcomes = row.get("plan_c_counterfactuals")
    if not isinstance(outcomes, list) or not outcomes:
        raise ValueError("counterfactual row has no sampled Plan C outcomes")
    for outcome in outcomes:
        descriptor = outcome.get("descriptor")
        if not isinstance(descriptor, dict):
            raise ValueError("sampled Plan C outcome has no exact descriptor")
        actual = _branch_return(scenario_id=row["scenario_id"],
            category=row["category"], seed=row["seed"],
            device=manifest["device"]["device"],
            horizon_ticks=manifest["horizon_physics_ticks"],
            action_ticks=manifest["max_macro_ticks"],
            decision_index=row["decision_index"], prefix=replay_prefix,
            descriptor=descriptor, kind="C",
            candidate_k=manifest["candidate_k"],
            prediction_mode=manifest["prediction_mode"],
            max_c_candidates=manifest["max_plan_c_descriptors_per_prefix"],
            max_a_pairs=manifest["max_plan_a_pairs_per_prefix"],
            gamma_per_macro=manifest["gamma_per_macro"])
        for field in expected_fields:
            if field not in outcome or outcome[field] != actual[field]:
                raise RuntimeError(
                    f"sampled branch outcome replay mismatch for {descriptor.get('candidate_id')}: {field}")


def analyze_sampled_regret(dataset: Path, checkpoint: Path, output: Path, *,
        track: str, qualification_protocol: Path, device: str):
    """Validate, replay, score, and write bounded sampled-set regret records."""
    if track not in ("C1", "C2"):
        raise ValueError("sampled counterfactual regret supports only C1 or C2")
    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise ValueError("an explicit indexed CUDA/ROCm device is required")
    sources = source_identity()
    rows, data_sha, manifest_sha, _, protocol_id, protocol_sha, dataset_manifest = \
        _read_counterfactual_dataset(Path(dataset),
            qualification_protocol=Path(qualification_protocol),
            expected_sources=sources)
    if dataset_manifest.get("schema") != "meco-counterfactual-trajectory-v3":
        raise ValueError("input is not replay-bound counterfactual-v2 trajectory data")
    if dataset_manifest.get("prediction_mode") not in PREDICTION_MODES:
        raise ValueError("dataset prediction mode is invalid")
    model, checkpoint_payload, config = _load_c_checkpoint(track, Path(checkpoint),
        manifest=dataset_manifest, expected_sources=sources)
    device_obj = torch.device(device)
    if not torch.cuda.is_available() or device_obj.index is None or device_obj.index >= torch.cuda.device_count():
        raise ValueError("requested indexed CUDA/ROCm device is unavailable")
    model = model.to(device_obj)
    checkpoint_sha = _file_sha256(Path(checkpoint))
    checkpoint_config_sha = _config_digest(config)

    # This shared validator reconstructs the candidate table and verifies every
    # sampled descriptor/outcome against strict branch replay before inference.
    materialized = _materialize_counterfactual_rows(rows, dataset_manifest,
        track=track, train_device=device)
    if len(materialized) != len(rows):
        raise RuntimeError("replayed materialization row count differs from dataset")
    output_rows = []
    reason_counts = {}
    regret_values = []
    candidate_ids_seen = 0
    candidate_ids_chosen = 0

    for row, sample in zip(rows, materialized):
        table = sample["table"]
        if sample["kind"] != "C":
            raise ValueError("Plan A row found in Plan C regret input")
        # Recreate hidden state by scoring every exact teacher-prefix table,
        # then use it to score this branch's exact replayed candidate table.
        logits, replay_prefix = _replay_teacher_prefix_and_score(row,
            dataset_manifest, sample, model, track=track, device=device)
        _verify_sampled_branch_outcomes(row, dataset_manifest, replay_prefix)
        # argmax preserves the production scorer's stable first-row tie rule.
        chosen_index = int(torch.argmax(logits).item())
        if not table.valid_mask[chosen_index]:
            raise RuntimeError("masked candidate scorer selected an invalid row")
        chosen_id = table.candidate_ids[chosen_index]
        outcome_by_id = {}
        for candidate_index, gross_return, candidate_id in sample["pairs"]:
            if candidate_id in outcome_by_id:
                raise ValueError("duplicate exact candidate ID in sampled outcomes")
            if table.candidate_ids[candidate_index] != candidate_id:
                raise ValueError("sample outcome candidate ID differs from exact replay table")
            outcome_by_id[candidate_id] = float(gross_return)
        candidate_ids_seen += len(table.candidate_ids)
        candidate_ids_chosen += 1
        sampled_returns = list(outcome_by_id.values())
        max_return = max(sampled_returns) if sampled_returns else None
        chosen_available = chosen_id in outcome_by_id
        chosen_return = outcome_by_id.get(chosen_id)
        if not chosen_available:
            excluded_reason = "checkpoint_chosen_candidate_outcome_not_sampled"
            reason_counts[excluded_reason] = reason_counts.get(excluded_reason, 0) + 1
            regret = None
        else:
            excluded_reason = None
            regret = max_return - chosen_return
            # Floating returns originate from the same stored values; a
            # negative result indicates an indexing/provenance bug.
            if regret < 0.:
                raise RuntimeError("computed sampled-set regret is negative")
            regret_values.append(regret)
        if not sampled_returns:
            no_outcomes = "no_sampled_candidate_outcomes"
            reason_counts[no_outcomes] = reason_counts.get(no_outcomes, 0) + 1
            if excluded_reason is None:
                raise RuntimeError("chosen outcome exists despite empty sampled set")
        output_rows.append({
            "scenario_id": row["scenario_id"], "category": row["category"],
            "seed": row["seed"], "decision_index": row["decision_index"],
            "snapshot_id": row["branch_state"]["snapshot_id"],
            "candidate_digest": row["candidate_digest"],
            "candidate_replay_digest": row["candidate_replay_digest"],
            "support_digest": row["support_digest"],
            "candidate_table_order_sha256": _canonical_sha256(list(table.candidate_ids)),
            "sampled_candidate_ids": list(outcome_by_id),
            "sampled_candidate_support_sha256": _canonical_sha256(sorted(outcome_by_id)),
            "valid_candidate_count": sum(bool(value) for value in table.valid_mask),
            "sampled_outcome_count": len(outcome_by_id),
            "chosen_candidate_id": chosen_id,
            "chosen_candidate_outcome_available": chosen_available,
            "sampled_max_gross_horizon_return": max_return,
            "chosen_gross_horizon_return": chosen_return,
            "sampled_bounded_set_regret": regret,
            "excluded_reason": excluded_reason,
        })

    eligible_count = len(output_rows)
    covered_count = len(regret_values)
    report = {"schema": REPORT_SCHEMA,
        "label": "sampled bounded-set regret; not simulator/global optimality",
        "regret_semantics": REGRET_SEMANTICS,
        "scoring_semantics": SCORING_SEMANTICS,
        "track": track,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(dataset).resolve()), "dataset_sha256": data_sha,
        "dataset_manifest_sha256": manifest_sha,
        "dataset_schema": dataset_manifest["schema"],
        "target_schema": dataset_manifest["target_schema"],
        "source_identity_sha256": dataset_manifest["source_identity_sha256"],
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_config_sha256": checkpoint_config_sha,
        "prediction_mode": dataset_manifest["prediction_mode"],
        "scoring_device": device,
        "prediction_contract": dataset_manifest["prediction_contract"],
        "candidate_k": dataset_manifest["candidate_k"],
        "candidate_boundary_sha256": _canonical_sha256(dataset_manifest["candidate_boundary"]),
        "route_cost_points_per_second": dataset_manifest["route_cost_points_per_second"],
        "risk_cost_semantics": "exact replayed risk tensor for the dataset prediction mode",
        "control_comparisons": {
            "D1": {"status": "unavailable",
                "reason": "this analyzer computes C-checkpoint sampled regret only; the source table contains one mode-specific risk estimate"},
            "D2": {"status": "unavailable",
                "reason": "this analyzer computes C-checkpoint sampled regret only; the source table contains one mode-specific risk estimate"},
            "note": "No D1/D2 regret or contrast is computed; a shared archived risk tensor is not reported as two independent controls"},
        "candidate_support_digest_semantics": "per-row digest covers exact sampled candidate IDs; full replayed candidate order digest also recorded",
        "coverage": {"branch_points": eligible_count,
            "chosen_outcome_available": covered_count,
            "chosen_outcome_missing": eligible_count - covered_count,
            "chosen_outcome_coverage_fraction": (covered_count / eligible_count if eligible_count else None),
            "candidate_ids_scored": candidate_ids_seen,
            "candidate_decisions_scored": candidate_ids_chosen,
            "excluded_reason_counts": reason_counts},
        "sampled_bounded_set_regret": {
            "count": covered_count,
            "mean": sum(regret_values) / covered_count if covered_count else None,
            "max": max(regret_values) if regret_values else None,
            "min": min(regret_values) if regret_values else None},
        "records": output_rows}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return {"path": str(output), "schema": REPORT_SCHEMA,
        "branch_points": eligible_count, "covered": covered_count,
        "mean_sampled_bounded_set_regret": report["sampled_bounded_set_regret"]["mean"],
        "dataset_sha256": data_sha, "checkpoint_sha256": checkpoint_sha}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("C1", "C2"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification-protocol", type=Path, required=True)
    parser.add_argument("--device", required=True,
        help="explicit indexed CUDA/ROCm device, for example cuda:0")
    args = parser.parse_args(argv)
    result = analyze_sampled_regret(args.dataset, args.checkpoint, args.output,
        track=args.track, qualification_protocol=args.qualification_protocol,
        device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
