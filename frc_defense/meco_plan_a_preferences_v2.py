"""Strict intake for human-authored, replay-bound Plan A goal preferences v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from . import meco_counterfactual as counterfactual
from . import meco_plan_a_annotations as annotations
from . import meco_supervised as supervised

SCHEMA = "meco-plan-a-human-preferences-v2"
TARGET_SCHEMA = "meco-plan-a-pairwise-goal-and-behavior-preferences-v2"
IDENTITY = annotations.IDENTITY
LABEL_KEYS = set(IDENTITY) | {"goal_preference_tiers", "behavior_preferences"}
ALLOWED_LABEL_KEYS = LABEL_KEYS | {"action_pair_preferences"}
SIGNED_CHOICE_SCHEMA = "meco-plan-a-signed-action-choice-v1"


def _index(rows, label):
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} record must be an object")
        key = annotations._key(row)
        if key in result:
            raise ValueError(f"duplicate {label} identity: {key}")
        result[key] = row
    return result


def _int_id(value):
    return type(value) is int and value >= 0


def _protocol_trusted_keys(protocol):
    """Read pinned Ed25519 trust anchors from the hashed qualification protocol."""
    registry = protocol.get("plan_a_expert_authority_keys") if isinstance(protocol, dict) else None
    if not isinstance(registry, dict) or not registry:
        raise ValueError("qualification protocol has no pinned Plan A expert authority keys")
    try:
        keys = {key_id: bytes.fromhex(encoded) for key_id, encoded in registry.items()
            if isinstance(key_id, str) and key_id and isinstance(encoded, str)}
    except ValueError as exc:
        raise ValueError("protocol-pinned public keys must be Ed25519 raw-key hex") from exc
    if len(keys) != len(registry) or any(len(key) != 32 for key in keys.values()):
        raise ValueError("protocol-pinned Ed25519 key registry is malformed")
    return keys


def _validate_labels(label, key, legal):
    if (legal.get("schema") != counterfactual.PLAN_A_LEGAL_NODE_DATASET_SCHEMA or
            legal.get("target_schema") != counterfactual.PLAN_A_LEGAL_NODE_TARGET_SCHEMA or
            annotations._key(legal) != key):
        raise ValueError("legal-node record schema or identity mismatch")
    unknown = set(label) - ALLOWED_LABEL_KEYS
    if unknown:
        raise ValueError(f"unsupported preference fields: {sorted(unknown)}")
    tiers = label.get("goal_preference_tiers")
    raw_action_preferences = label.get("action_pair_preferences", [])
    if (not isinstance(tiers, list) or
            (len(tiers) < 2 and not raw_action_preferences) or
            any(not isinstance(tier, list) or not tier for tier in tiers)):
        raise ValueError("goal preferences or signed action-pair preferences are required")
    flattened = [node for tier in tiers for node in tier]
    if any(not _int_id(node) for node in flattened) or len(set(flattened)) != len(flattened):
        raise ValueError("goal preference tiers require unique nonnegative integer node IDs")
    raw_feasible = legal.get("feasible_node_ids")
    raw_poses = legal.get("canonical_poses")
    if (not isinstance(raw_feasible, list) or any(not _int_id(node) for node in raw_feasible) or
            len(set(raw_feasible)) != len(raw_feasible) or not isinstance(raw_poses, list)):
        raise ValueError("replayed legal-node support is malformed")
    feasible = set(raw_feasible)
    poses = {pose.get("node_id"): pose for pose in raw_poses if isinstance(pose, dict)}
    if (len(poses) != len(raw_poses) or
            any(not _int_id(node) or not isinstance(pose.get("x_m"), (int, float)) or
                isinstance(pose.get("x_m"), bool) or not math.isfinite(pose["x_m"]) or
                not isinstance(pose.get("y_m"), (int, float)) or isinstance(pose.get("y_m"), bool) or
                not math.isfinite(pose["y_m"]) or not _int_id(pose.get("heading_bin"))
                for node, pose in poses.items())):
        raise ValueError("canonical legal-node poses are malformed")
    if not set(flattened) <= feasible or not set(flattened) <= set(poses):
        raise ValueError("preferred goal is outside replayed feasible canonical node support")
    goal_preferences = []
    for left_rank, left_tier in enumerate(tiers):
        for right_tier in tiers[left_rank + 1:]:
            for left in left_tier:
                for right in right_tier:
                    goal_preferences.append({"left_node_id": left, "right_node_id": right,
                        "preferred_node_id": left})
    behavior_preferences = label.get("behavior_preferences")
    if not isinstance(behavior_preferences, list):
        raise ValueError("behavior_preferences must be an explicit list")
    pairs = {(pair.get("node_id"), pair.get("behavior_id"))
        for pair in legal.get("pairs", []) if isinstance(pair, dict)
        and pair.get("status") == "available"}
    seen = set()
    normalized = []
    for pref in behavior_preferences:
        if not isinstance(pref, dict) or set(pref) != {
                "node_id", "left_behavior_id", "right_behavior_id", "preferred_behavior_id"}:
            raise ValueError("behavior preference must name one supported pair and its preferred behavior")
        node, left, right, preferred = (pref[k] for k in (
            "node_id", "left_behavior_id", "right_behavior_id", "preferred_behavior_id"))
        if (not _int_id(node) or not _int_id(left) or not _int_id(right) or
                left == right or preferred not in (left, right) or
                (node, left) not in pairs or (node, right) not in pairs):
            raise ValueError("behavior preference is infeasible or unavailable in exact legal support")
        identity = (node, min(left, right), max(left, right))
        if identity in seen:
            raise ValueError("duplicate behavior preference pair")
        seen.add(identity)
        normalized.append({"node_id": node, "left_behavior_id": left,
            "right_behavior_id": right, "preferred_behavior_id": preferred})
    action_preferences = []
    seen_actions = set()
    if not isinstance(raw_action_preferences, list):
        raise ValueError("action_pair_preferences must be an explicit list")
    for pref in raw_action_preferences:
        fields = {"left_node_id", "left_behavior_id", "right_node_id",
                  "right_behavior_id", "preferred_node_id", "preferred_behavior_id"}
        if not isinstance(pref, dict) or set(pref) != fields:
            raise ValueError("action-pair preference must name two choices and the preferred choice")
        left = (pref["left_node_id"], pref["left_behavior_id"])
        right = (pref["right_node_id"], pref["right_behavior_id"])
        preferred = (pref["preferred_node_id"], pref["preferred_behavior_id"])
        if (any(not _int_id(x) for x in (*left, *right, *preferred)) or left == right or
                preferred not in (left, right) or left not in pairs or right not in pairs or
                left[0] not in feasible or right[0] not in feasible or
                left[0] not in poses or right[0] not in poses):
            raise ValueError("action-pair preference is infeasible or unavailable in exact legal support")
        pair_identity = tuple(sorted((left, right)))
        if pair_identity in seen_actions:
            raise ValueError("duplicate action-pair preference")
        seen_actions.add(pair_identity)
        action_preferences.append({"left_node_id": left[0], "left_behavior_id": left[1],
            "right_node_id": right[0], "right_behavior_id": right[1],
            "preferred_node_id": preferred[0], "preferred_behavior_id": preferred[1]})
    return goal_preferences, normalized, action_preferences


def materialize_plan_a_preferences_v2(*, source_dataset: Path,
        legal_node_dataset: Path, qualification_protocol: Path,
        authority_json: Path, output: Path, authored_rows: Path | None = None,
        signed_choice_sidecar: Path | None = None):
    """Bind human ranking tiers to exact train-split source and legal support."""
    source_dataset, legal_node_dataset, qualification_protocol, authority_json = map(
        Path, (source_dataset, legal_node_dataset, qualification_protocol, authority_json))
    authored_rows = Path(authored_rows) if authored_rows is not None else None
    input_paths = ([ ("authored rows", authored_rows)] if authored_rows else [])
    if signed_choice_sidecar is not None:
        signed_choice_sidecar = Path(signed_choice_sidecar)
        input_paths.append(("signed choice sidecar", signed_choice_sidecar))
        input_paths.append(("signed choice authority", Path(str(signed_choice_sidecar) + ".authority.json")))
    input_paths += [
        ("source dataset", source_dataset),
        ("source manifest", source_dataset.with_suffix(source_dataset.suffix + ".manifest.json")),
        ("legal node dataset", legal_node_dataset),
        ("legal node manifest", legal_node_dataset.with_suffix(legal_node_dataset.suffix + ".manifest.json")),
        ("protocol", qualification_protocol), ("authority", authority_json)]
    output, manifest_path = annotations._preflight_destinations(output, input_paths)
    from .curriculum_training import source_identity
    sources = source_identity()
    source, source_sha, source_manifest_sha, protocol, protocol_id, protocol_sha, source_manifest = \
        supervised._read_counterfactual_dataset(source_dataset,
            qualification_protocol=qualification_protocol, expected_sources=sources)
    legal_bytes = legal_node_dataset.read_bytes()
    legal_sha = hashlib.sha256(legal_bytes).hexdigest()
    legal_manifest_path = legal_node_dataset.with_suffix(legal_node_dataset.suffix + ".manifest.json")
    legal_manifest_bytes = legal_manifest_path.read_bytes()
    legal_manifest = json.loads(legal_manifest_bytes)
    if not isinstance(legal_manifest, dict):
        raise ValueError("legal-node manifest must be an object")
    if (legal_manifest.get("schema") != counterfactual.PLAN_A_LEGAL_NODE_DATASET_SCHEMA or
            legal_manifest.get("target_schema") != counterfactual.PLAN_A_LEGAL_NODE_TARGET_SCHEMA or
            legal_manifest.get("dataset_sha256") != legal_sha or
            legal_manifest.get("source_dataset_sha256") != source_sha or
            legal_manifest.get("source_manifest_sha256") != source_manifest_sha or
            legal_manifest.get("protocol_id") != protocol_id or
            legal_manifest.get("qualification_protocol_sha256") != protocol_sha or
            legal_manifest.get("split") != "train" or
            legal_manifest.get("sources") != sources or
            legal_manifest.get("source_identity_sha256") != supervised._canonical_hash(sources) or
            legal_manifest.get("device") != source_manifest.get("device")):
        raise ValueError("legal-node/source/protocol/split binding mismatch")
    legal_rows = [json.loads(line) for line in legal_bytes.decode("utf-8").splitlines() if line.strip()]
    source_by_key, legal_by_key = _index(source, "source"), _index(legal_rows, "legal-node")
    if len(legal_rows) != legal_manifest.get("record_count") or set(source_by_key) != set(legal_by_key):
        raise ValueError("source/legal-node records do not match exact branch identities")
    authority_bytes = authority_json.read_bytes()
    authority = json.loads(authority_bytes)
    signed_manifest = None
    signed_choices = None
    if signed_choice_sidecar is not None:
        from . import meco_plan_a_expert_choice as choice_authority
        sidecar_bytes = Path(signed_choice_sidecar).read_bytes()
        signed_choices = annotations._jsonl(Path(signed_choice_sidecar))
        public_keys = _protocol_trusted_keys(protocol)
        signed_manifest = json.loads(Path(str(signed_choice_sidecar) + ".authority.json").read_bytes())
        payload = choice_authority.verify_authority(signed_manifest, trusted_keys=public_keys)
        expected_hashes = {"sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            "source_sha256": source_sha,
            "source_manifest_sha256": source_manifest_sha,
            "replay_sha256": legal_sha,
            "replay_manifest_sha256": hashlib.sha256(legal_manifest_bytes).hexdigest()}
        if any(payload.get(name) != value for name, value in expected_hashes.items()):
            raise ValueError("signed human choice source/replay hash binding mismatch")
        if sidecar_bytes != b"".join((json.dumps(row, sort_keys=True,
                separators=(",", ":"), allow_nan=False) + "\n").encode()
                for row in signed_choices):
            raise ValueError("signed human choice sidecar must use canonical JSONL encoding")
        choice_by_key = _index(signed_choices, "signed-choice")
        if set(choice_by_key) != set(source_by_key) or [annotations._key(row)
                for row in signed_choices] != sorted(source_by_key):
            raise ValueError("signed human choices must cover exact source train identities")
        authored = {}
        for key, choice in choice_by_key.items():
            src, legal = source_by_key[key], legal_by_key[key]
            expected_state = src.get("predecision_simulator_state_sha256")
            expected_snapshot = src.get("branch_state", {}).get("snapshot_id")
            if (choice.get("schema") != SIGNED_CHOICE_SCHEMA or
                    choice.get("snapshot_id") != expected_snapshot or
                    choice.get("state_sha256") != expected_state or
                    choice.get("support_digest") != legal.get("support_digest")):
                raise ValueError("signed choice does not bind exact source state and legal support")
            selected = (choice.get("node_id"), choice.get("behavior_id"))
            baseline = (choice.get("baseline_node_id"), choice.get("baseline_behavior_id"))
            available = {(pair.get("node_id"), pair.get("behavior_id"))
                for pair in legal.get("pairs", []) if pair.get("status") == "available"}
            if (any(not _int_id(value) for value in (*selected, *baseline)) or
                    selected == baseline or selected not in available or baseline not in available):
                raise ValueError("signed choice and comparison baseline must be distinct feasible support pairs")
            authored[key] = {**dict(zip(IDENTITY, key)),
                # A complete-action choice does not independently identify a
                # U-only goal ranking; keep its supervision in the joint U+B
                # action-pair target below.
                "goal_preference_tiers": [], "behavior_preferences": [],
                "action_pair_preferences": [{
                    "left_node_id": selected[0], "left_behavior_id": selected[1],
                    "right_node_id": baseline[0], "right_behavior_id": baseline[1],
                    "preferred_node_id": selected[0], "preferred_behavior_id": selected[1]}]}
        authority_source_id = payload["author_id"]
    else:
        if authored_rows is None:
            raise ValueError("authored_rows is required without a signed human-choice sidecar")
        if (not isinstance(authority, dict) or not isinstance(authority.get("source_id"), str)
                or not authority["source_id"].strip()):
            raise ValueError("authority JSON must declare a nonempty source_id")
        authority_source_id = authority["source_id"]
        authored = _index(annotations._jsonl(authored_rows), "authored")
    if not authored:
        raise ValueError("preference annotation dataset is empty")
    if not set(authored) <= set(source_by_key):
        raise ValueError("authored identities are outside the exact source train dataset")
    rows = []
    for key in sorted(authored):
        label, src, legal = authored[key], source_by_key[key], legal_by_key[key]
        if set(label) not in (LABEL_KEYS, ALLOWED_LABEL_KEYS):
            raise ValueError("authored preference row has unsupported or missing fields")
        branch_state = src.get("branch_state")
        state_sha = src.get("predecision_simulator_state_sha256")
        snapshot_id = branch_state.get("snapshot_id") if isinstance(branch_state, dict) else None
        if (src.get("schema") != counterfactual.DATASET_SCHEMA or
                src.get("target_schema") != counterfactual.TARGET_SCHEMA or
                not isinstance(snapshot_id, str) or not snapshot_id or
                not isinstance(state_sha, str) or len(state_sha) != 64 or
                any(char not in "0123456789abcdef" for char in state_sha) or
                not isinstance(src.get("support_digest"), str) or len(src["support_digest"]) != 64 or
                any(char not in "0123456789abcdef" for char in src["support_digest"])):
            raise ValueError("counterfactual row lacks valid snapshot, state, or support provenance")
        if (legal.get("snapshot_id") != snapshot_id or
                legal.get("state_sha256") != state_sha or
                legal.get("support_digest") != src.get("support_digest")):
            raise ValueError("source/legal-node rows do not bind the same snapshot and support digest")
        if key[2] not in supervised._protocol_train_seeds(protocol, key[0], key[1]):
            raise ValueError(f"preference annotation uses heldout or undeclared seed: {key}")
        goal_prefs, behavior_prefs, action_prefs = _validate_labels(label, key, legal)
        rows.append({"schema": SCHEMA, "target_schema": TARGET_SCHEMA,
            **dict(zip(IDENTITY, key)), "annotation_source_id": authority_source_id,
            "snapshot_id": legal["snapshot_id"], "state_sha256": legal["state_sha256"],
            "support_digest": legal["support_digest"],
            "goal_preference_tiers": label["goal_preference_tiers"],
            "goal_preferences": goal_prefs,
            "behavior_preferences": behavior_prefs,
            "action_pair_preferences": action_prefs,
            **({"signed_human_choice": choice_by_key[key]}
               if signed_choice_sidecar is not None else {}),
            "u_numeric_status": "unavailable", "u_numeric_target": None})
    data = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"),
        allow_nan=False) + "\n").encode() for row in rows)
    manifest = {"schema": SCHEMA, "target_schema": TARGET_SCHEMA,
        "target_semantics": "human_pairwise_goal_ranking_and_behavior_preferences; no_numeric_u",
        "split": "train", "source_dataset_sha256": source_sha,
        "source_manifest_sha256": source_manifest_sha,
        "legal_node_dataset_sha256": legal_sha,
        "legal_node_manifest_sha256": hashlib.sha256(legal_manifest_bytes).hexdigest(),
        "protocol_id": protocol_id, "qualification_protocol_sha256": protocol_sha,
        "sources": sources, "source_identity_sha256": supervised._canonical_hash(sources),
        "dataset_sha256": hashlib.sha256(data).hexdigest(),
        "annotation_source": {"source_id": authority_source_id,
            "sha256": hashlib.sha256(authority_bytes).hexdigest()},
        "signed_human_choice_authority": signed_manifest,
        "signed_human_choice_sidecar_sha256": (hashlib.sha256(
            Path(signed_choice_sidecar).read_bytes()).hexdigest()
            if signed_choice_sidecar is not None else None),
        "authority_declared_only": signed_manifest is None,
        "authority_is_authenticated_identity": signed_manifest is not None,
        "authority_proves_expertise_or_optimality": False,
        "replay_validation_required": True, "record_count": len(rows),
        "authored_rows_sha256": hashlib.sha256((Path(signed_choice_sidecar).read_bytes()
            if signed_choice_sidecar is not None else authored_rows.read_bytes())).hexdigest(),
        "u_numeric_target": None,
        "u_numeric_unavailable_reason": "no independent complete uncensored oracle values supplied",
        "p_target": None, "p_mode": "p0"}
    manifest_data = (json.dumps(manifest, sort_keys=True, separators=(",", ":"),
        allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    annotations._publish_pair(output, manifest_path, data, manifest_data)
    return {"dataset": str(output), "rows": len(rows),
        "dataset_sha256": hashlib.sha256(data).hexdigest(), "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
        "replay_validation_required": True}


def load_plan_a_preferences_v2(dataset: Path, legal_node_dataset: Path,
        source_dataset: Path, *, qualification_protocol: Path,
        expected_sources: dict, authority_json: Path, device: str, track=None):
    """Validate labels and regenerate their legal support from simulator replay."""
    dataset, legal_node_dataset, source_dataset = map(Path.resolve,
        (Path(dataset), Path(legal_node_dataset), Path(source_dataset)))
    manifest_path = dataset.with_suffix(dataset.suffix + ".manifest.json")
    manifest_bytes, data_bytes = manifest_path.read_bytes(), dataset.read_bytes()
    manifest = json.loads(manifest_bytes)
    legal_path = legal_node_dataset.with_suffix(legal_node_dataset.suffix + ".manifest.json")
    legal_bytes, legal_manifest_bytes = legal_node_dataset.read_bytes(), legal_path.read_bytes()
    legal_manifest = json.loads(legal_manifest_bytes)
    if not isinstance(manifest, dict) or not isinstance(legal_manifest, dict):
        raise ValueError("preference and legal-node manifests must be objects")
    protocol, protocol_id, protocol_sha, _ = supervised._load_protocol(qualification_protocol)
    source, source_sha, source_manifest_sha, _, source_protocol_id, source_protocol_sha, source_manifest = \
        supervised._read_counterfactual_dataset(source_dataset,
            qualification_protocol=qualification_protocol, expected_sources=expected_sources)
    data_sha, legal_sha = hashlib.sha256(data_bytes).hexdigest(), hashlib.sha256(legal_bytes).hexdigest()
    authority_bytes = Path(authority_json).resolve().read_bytes()
    authority = json.loads(authority_bytes)
    if not isinstance(authority, dict):
        raise ValueError("authority JSON must be an object")
    annotation_source = manifest.get("annotation_source")
    signed_authority = manifest.get("signed_human_choice_authority")
    if (manifest.get("schema") != SCHEMA or manifest.get("target_schema") != TARGET_SCHEMA or
            manifest.get("split") != "train" or manifest.get("dataset_sha256") != data_sha or
            manifest.get("source_dataset_sha256") != source_sha or
            manifest.get("source_manifest_sha256") != source_manifest_sha or
            manifest.get("legal_node_dataset_sha256") != legal_sha or
            manifest.get("legal_node_manifest_sha256") != hashlib.sha256(legal_manifest_bytes).hexdigest() or
            manifest.get("protocol_id") != protocol_id or manifest.get("protocol_id") != source_protocol_id or
            manifest.get("qualification_protocol_sha256") != protocol_sha or
            manifest.get("qualification_protocol_sha256") != source_protocol_sha or
            manifest.get("sources") != expected_sources or
            manifest.get("source_identity_sha256") != supervised._canonical_hash(expected_sources) or
            manifest.get("u_numeric_target") is not None or
            manifest.get("u_numeric_unavailable_reason") !=
                "no independent complete uncensored oracle values supplied" or
            not isinstance(authority, dict) or not isinstance(annotation_source, dict) or
            hashlib.sha256(authority_bytes).hexdigest() != annotation_source.get("sha256") or
            (signed_authority is None and
             authority.get("source_id") != annotation_source.get("source_id"))):
        raise ValueError("preference annotation manifest/source/protocol/authority binding mismatch")
    if (legal_manifest.get("schema") != counterfactual.PLAN_A_LEGAL_NODE_DATASET_SCHEMA or
            legal_manifest.get("target_schema") != counterfactual.PLAN_A_LEGAL_NODE_TARGET_SCHEMA or
            legal_manifest.get("split") != "train" or
            legal_manifest.get("dataset_sha256") != legal_sha or
            legal_manifest.get("source_dataset_sha256") != source_sha or
            legal_manifest.get("source_manifest_sha256") != source_manifest_sha or
            legal_manifest.get("protocol_id") != protocol_id or
            legal_manifest.get("qualification_protocol_sha256") != protocol_sha or
            legal_manifest.get("sources") != expected_sources or
            legal_manifest.get("source_identity_sha256") != supervised._canonical_hash(expected_sources)):
        raise ValueError("legal-node dataset is not bound to the exact source/protocol/train split")
    legal_rows = _index([json.loads(line) for line in legal_bytes.decode().splitlines() if line.strip()],
        "legal-node")
    source_rows = _index(source, "source")
    if (len(legal_rows) != legal_manifest.get("record_count") or
            set(legal_rows) != set(source_rows)):
        raise ValueError("source/legal-node row identities differ")
    rows = annotations._index(annotations._jsonl(dataset), "preference")
    if not rows or len(rows) != manifest.get("record_count"):
        raise ValueError("preference annotation record count is invalid")
    if signed_authority is not None:
        from . import meco_plan_a_expert_choice as choice_authority
        trusted_keys = _protocol_trusted_keys(protocol)
        payload = choice_authority.verify_authority(signed_authority,
            trusted_keys=trusted_keys)
        signed_rows = [rows[key].get("signed_human_choice") for key in sorted(rows)]
        if any(not isinstance(item, dict) for item in signed_rows):
            raise ValueError("signed choice rows are missing from preference dataset")
        reconstructed_sidecar = b"".join((json.dumps(item, sort_keys=True,
            separators=(",", ":"), allow_nan=False) + "\n").encode()
            for item in signed_rows)
        expected_hashes = {"sidecar_sha256": hashlib.sha256(reconstructed_sidecar).hexdigest(),
            "source_sha256": source_sha,
            "source_manifest_sha256": source_manifest_sha,
            "replay_sha256": legal_sha,
            "replay_manifest_sha256": hashlib.sha256(legal_manifest_bytes).hexdigest()}
        if (manifest.get("signed_human_choice_sidecar_sha256") != expected_hashes["sidecar_sha256"] or
                payload.get("author_id") != annotation_source.get("source_id") or
                any(payload.get(name) != value for name, value in expected_hashes.items())):
            raise ValueError("signed human choice authority/source/replay binding mismatch")
        expected_author_id = payload["author_id"]
    else:
        expected_author_id = authority.get("source_id")
    replayed, replay_manifest, replay_legal_sha, replay_manifest_sha = \
        supervised._read_and_replay_plan_a_legal_node_dataset(legal_node_dataset, source_dataset,
            qualification_protocol, device, track)
    if replay_legal_sha != legal_sha or replay_manifest_sha != hashlib.sha256(legal_manifest_bytes).hexdigest():
        raise ValueError("legal-node dataset changed during fresh replay validation "
            f"(data {replay_legal_sha}/{legal_sha}; manifest {replay_manifest_sha}/"
            f"{hashlib.sha256(legal_manifest_bytes).hexdigest()})")
    replay_by_key = {(sample["row"]["scenario_id"], sample["row"]["category"],
        sample["row"]["seed"], sample["row"]["decision_index"]): sample for sample in replayed}
    if set(rows) - set(replay_by_key):
        raise ValueError("preference annotations contain identities outside replayed support")
    validated = []
    for key, row in rows.items():
        sample = replay_by_key[key]
        legal = sample["row"]
        signed_choice = row.get("signed_human_choice")
        authored = {**dict(zip(IDENTITY, key)),
            "goal_preference_tiers": row.get("goal_preference_tiers"),
            "behavior_preferences": row.get("behavior_preferences"),
            "action_pair_preferences": row.get("action_pair_preferences")}
        goal_prefs, behavior_prefs, action_prefs = _validate_labels(authored, key, legal)
        if (row.get("schema") != SCHEMA or row.get("target_schema") != TARGET_SCHEMA or
                row.get("annotation_source_id") != expected_author_id or
                row.get("snapshot_id") != legal.get("snapshot_id") or
                row.get("state_sha256") != legal.get("state_sha256") or
                row.get("support_digest") != legal.get("support_digest") or
                row.get("goal_preferences") != goal_prefs or
                row.get("behavior_preferences") != behavior_prefs or
                row.get("action_pair_preferences") != action_prefs or
                row.get("u_numeric_status") != "unavailable" or
                row.get("u_numeric_target") is not None or
                key[2] not in supervised._protocol_train_seeds(protocol, key[0], key[1])):
            raise ValueError("preference row is malformed, held out, or differs from exact replay support")
        if signed_authority is not None:
            from .meco_counterfactual import _select_plan_a_expert_goal_v1
            if (not isinstance(signed_choice, dict) or
                    signed_choice.get("schema") != SIGNED_CHOICE_SCHEMA or
                    signed_choice.get("scenario_id") != key[0] or
                    signed_choice.get("category") != key[1] or
                    signed_choice.get("seed") != key[2] or
                    signed_choice.get("decision_index") != key[3] or
                    signed_choice.get("snapshot_id") != legal.get("snapshot_id") or
                    signed_choice.get("state_sha256") != legal.get("state_sha256") or
                    signed_choice.get("support_digest") != sample["support"].support_digest):
                raise ValueError("signed choice does not bind the freshly replayed legal support")
            _, baseline = _select_plan_a_expert_goal_v1(sample["support"])
            if ((signed_choice.get("baseline_node_id"),
                    signed_choice.get("baseline_behavior_id")) !=
                    (baseline["node_id"], baseline["behavior_id"]) or
                    (signed_choice.get("node_id"), signed_choice.get("behavior_id")) ==
                    (baseline["node_id"], baseline["behavior_id"])):
                raise ValueError("signed human choice is not distinct from exact min-route baseline")
        validated.append((row, sample))
    return validated, manifest, data_sha, hashlib.sha256(manifest_bytes).hexdigest(), replay_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authored-rows", type=Path)
    parser.add_argument("--signed-choice-sidecar", type=Path)
    for name in ("source-dataset", "legal-node-dataset", "qualification-protocol",
            "authority-json", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.authored_rows is None) == (args.signed_choice_sidecar is None):
        parser.error("provide exactly one of --authored-rows or --signed-choice-sidecar")
    print(json.dumps(materialize_plan_a_preferences_v2(authored_rows=args.authored_rows,
        source_dataset=args.source_dataset, legal_node_dataset=args.legal_node_dataset,
        qualification_protocol=args.qualification_protocol, authority_json=args.authority_json,
        output=args.output, signed_choice_sidecar=args.signed_choice_sidecar), sort_keys=True))


if __name__ == "__main__":
    main()
