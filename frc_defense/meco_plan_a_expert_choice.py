"""Signed human Plan A choices, kept separate from the minimum-route baseline.

This module validates a sidecar against frozen legal support and replay evidence.
It deliberately does not synthesize expert labels or treat the route baseline as
human authority.
"""
from __future__ import annotations

import hashlib
import json

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


SCHEMA = "meco-plan-a-human-choice-authority-v1"


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def authority_payload(*, key_id, author_id, basis, sidecar_sha256,
                      source_sha256, source_manifest_sha256, replay_sha256,
                      replay_manifest_sha256):
    """Canonical signed object; all provenance hashes are mandatory."""
    values = {"schema": SCHEMA, "key_id": key_id, "author_id": author_id,
        "basis": basis, "sidecar_sha256": sidecar_sha256,
        "source_sha256": source_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "replay_sha256": replay_sha256,
        "replay_manifest_sha256": replay_manifest_sha256}
    if any(not isinstance(v, str) or not v for v in values.values()):
        raise ValueError("expert choice authority fields must be nonempty strings")
    for key in ("sidecar_sha256", "source_sha256", "source_manifest_sha256",
                "replay_sha256", "replay_manifest_sha256"):
        value = values[key]
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("expert choice authority contains an invalid SHA-256")
    return values


def verify_authority(manifest, *, trusted_keys):
    """Verify Ed25519 signature against a caller-pinned key-id registry."""
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("unknown expert choice authority schema")
    payload = manifest.get("payload")
    if not isinstance(payload, dict) or payload.get("key_id") not in trusted_keys:
        raise ValueError("untrusted expert choice authority key")
    if payload.get("schema") != SCHEMA:
        raise ValueError("invalid expert choice authority payload")
    try:
        signature = bytes.fromhex(manifest["signature_ed25519"])
        key = Ed25519PublicKey.from_public_bytes(trusted_keys[payload["key_id"]])
        key.verify(signature, canonical_bytes(payload))
    except Exception as exc:
        raise ValueError("invalid expert choice authority signature") from exc
    return payload


def validate_human_choice_source(*, sidecar_bytes, authority_manifest,
        trusted_keys, source_bytes, source_manifest_bytes, replay_bytes,
        replay_manifest_bytes, source_rows, legal_rows, replay_rows,
        split="train"):
    """Validate signed decisions and exact execution evidence, return labels.

    `legal_rows` and `replay_rows` must come from their existing replay-validating
    loaders. A replay row must name the executed pair and carry false failure,
    remask, target-loss, and preemption flags.
    """
    hashes = {"sidecar_sha256": sha256_bytes(sidecar_bytes),
        "source_sha256": sha256_bytes(source_bytes),
        "source_manifest_sha256": sha256_bytes(source_manifest_bytes),
        "replay_sha256": sha256_bytes(replay_bytes),
        "replay_manifest_sha256": sha256_bytes(replay_manifest_bytes)}
    payload = verify_authority(authority_manifest, trusted_keys=trusted_keys)
    for key, value in hashes.items():
        if payload.get(key) != value:
            raise ValueError("expert choice signed provenance hash mismatch")
    try:
        decisions = [json.loads(line) for line in sidecar_bytes.decode().splitlines() if line.strip()]
    except Exception as exc:
        raise ValueError("malformed expert choice sidecar") from exc
    if not decisions:
        raise ValueError("expert choice sidecar is empty")
    for blob, rows, label in ((source_bytes, source_rows, "source"),
                              (replay_bytes, replay_rows, "replay")):
        try:
            decoded = [json.loads(line) for line in blob.decode().splitlines() if line.strip()]
        except Exception as exc:
            raise ValueError(f"malformed bound {label} data") from exc
        if decoded != list(rows):
            raise ValueError(f"bound {label} bytes differ from validated rows")
    if split != "train":
        raise ValueError("human choice labels may only be fitted from train split")
    def keyed(rows, label):
        result = {}
        for row in rows:
            key = (row.get("scenario_id"), row.get("category"), row.get("seed"),
                   row.get("decision_index"))
            if key in result:
                raise ValueError(f"duplicate {label} decision identity")
            result[key] = row
        return result
    source = keyed(source_rows, "source")
    legal = keyed(legal_rows, "legal-support")
    replay = keyed(replay_rows, "replay")
    seen, output = set(), []
    for decision in decisions:
        key = (decision.get("scenario_id"), decision.get("category"),
               decision.get("seed"), decision.get("decision_index"))
        if key in seen:
            raise ValueError("duplicate human choice decision")
        seen.add(key)
        if key not in source or key not in legal or key not in replay:
            raise ValueError("human choice is missing source, support, or replay row")
        base, support, executed = source[key], legal[key], replay[key]
        if decision.get("snapshot_id") != base.get("snapshot_id") or \
                decision.get("state_sha256") != base.get("state_sha256") or \
                decision.get("support_digest") != support.get("support_digest"):
            raise ValueError("human choice state or support binding mismatch")
        node, behavior = decision.get("node_id"), decision.get("behavior_id")
        if type(node) is not int or type(behavior) is not int:
            raise ValueError("human choice pair must contain integer node and behavior IDs")
        if not any(p.get("node_id") == node and p.get("behavior_id") == behavior and
                   p.get("status") == "available" for p in support.get("pairs", [])):
            raise ValueError("human choice is infeasible or remasked in frozen support")
        pose = next((p for p in support.get("canonical_poses", [])
                     if p.get("node_id") == node), None)
        if pose is None:
            raise ValueError("human choice has no canonical supported pose")
        if (executed.get("node_id") != node or executed.get("behavior_id") != behavior or
                executed.get("support_digest") != decision["support_digest"] or
                executed.get("snapshot_id") != decision["snapshot_id"] or
                executed.get("state_sha256") != decision["state_sha256"]):
            raise ValueError("actual execution does not match signed human choice")
        for flag in ("execution_failure", "action_remasked", "selected_target_action_remasked",
                     "selected_target_lost", "native_goal_preempted"):
            if executed.get(flag) is not False:
                raise ValueError("human choice replay is failed, remasked, or preempted")
        output.append({**decision, "goal_pose": pose,
            "author_id": payload["author_id"], "authority_key_id": payload["key_id"],
            "choice_source": SCHEMA})
    expected = set(source)
    if seen != expected:
        raise ValueError("human choice source has missing or extra decision rows")
    return output
