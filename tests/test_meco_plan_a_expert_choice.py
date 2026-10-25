import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from frc_defense import meco_plan_a_expert_choice as authority


def _fixture(tmp_path):
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    source = [{"scenario_id": "s", "category": "c", "seed": 12,
        "decision_index": 0, "snapshot_id": "snap", "state_sha256": "a" * 64}]
    legal = [{"scenario_id": "s", "category": "c", "seed": 12,
        "decision_index": 0, "support_digest": "b" * 64,
        "canonical_poses": [{"node_id": 1, "x_m": 0.0, "y_m": 1.0,
                             "heading_bin": 2}],
        "pairs": [{"node_id": 0, "behavior_id": 0, "status": "available"},
                  {"node_id": 1, "behavior_id": 2, "status": "available"}]}]
    replay = [{"scenario_id": "s", "category": "c", "seed": 12,
        "decision_index": 0, "snapshot_id": "snap", "state_sha256": "a" * 64,
        "support_digest": "b" * 64, "node_id": 1, "behavior_id": 2,
        "execution_failure": False, "action_remasked": False,
        "selected_target_action_remasked": False, "selected_target_lost": False,
        "native_goal_preempted": False}]
    decisions = [{"scenario_id": "s", "category": "c", "seed": 12,
        "decision_index": 0, "snapshot_id": "snap", "state_sha256": "a" * 64,
        "support_digest": "b" * 64, "node_id": 1, "behavior_id": 2}]
    def jsonl(rows):
        return b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                       for row in rows)
    sidecar, source_bytes, replay_bytes = jsonl(decisions), jsonl(source), jsonl(replay)
    blobs = [sidecar, source_bytes, b"source-manifest", replay_bytes, b"replay-manifest"]
    payload = authority.authority_payload(key_id="reviewer-key", author_id="human-7",
        basis="operator-selected feasible preference", sidecar_sha256=authority.sha256_bytes(blobs[0]),
        source_sha256=authority.sha256_bytes(blobs[1]),
        source_manifest_sha256=authority.sha256_bytes(blobs[2]),
        replay_sha256=authority.sha256_bytes(blobs[3]),
        replay_manifest_sha256=authority.sha256_bytes(blobs[4]))
    signed = {"schema": authority.SCHEMA, "payload": payload,
              "signature_ed25519": private.sign(authority.canonical_bytes(payload)).hex()}
    args = dict(sidecar_bytes=sidecar, authority_manifest=signed,
        trusted_keys={"reviewer-key": public}, source_bytes=blobs[1],
        source_manifest_bytes=blobs[2], replay_bytes=blobs[3],
        replay_manifest_bytes=blobs[4], source_rows=source, legal_rows=legal,
        replay_rows=replay)
    return args, decisions


def test_signed_human_choice_may_differ_from_min_route_baseline(tmp_path):
    args, _ = _fixture(tmp_path)
    # The deterministic baseline pair here is node 0 / behavior 0.
    result = authority.validate_human_choice_source(**args)
    assert (result[0]["node_id"], result[0]["behavior_id"]) == (1, 2)
    assert result[0]["goal_pose"]["heading_bin"] == 2
    assert result[0]["choice_source"] == authority.SCHEMA


@pytest.mark.parametrize("mutation, message", [
    ("sidecar", "signed provenance hash mismatch"),
    ("signature", "invalid expert choice authority signature"),
    ("untrusted", "untrusted expert choice authority key"),
    ("infeasible", "infeasible or remasked"),
    ("remasked", "infeasible or remasked"),
    ("preempted", "failed, remasked, or preempted"),
    ("mismatch", "actual execution does not match"),
    ("duplicate", "duplicate human choice"),
    ("missing", "sidecar is empty"),
    ("heldout", "only be fitted from train split"),
])
def test_invalid_authority_or_rows_are_rejected(tmp_path, mutation, message):
    args, decisions = _fixture(tmp_path)
    if mutation == "sidecar":
        args["sidecar_bytes"] += b" "
    elif mutation == "signature":
        args["authority_manifest"]["signature_ed25519"] = "00" * 64
    elif mutation == "untrusted":
        args["trusted_keys"] = {}
    elif mutation == "infeasible":
        decisions[0]["behavior_id"] = 4
        args["sidecar_bytes"] = (json.dumps(decisions[0], sort_keys=True,
            separators=(",", ":")) + "\n").encode()
        # Re-sign to ensure the failure is support feasibility, not tampering.
        args, _ = _fixture(tmp_path)
        row = json.loads(args["sidecar_bytes"])
        row["node_id"] = 999
        args["sidecar_bytes"] = (json.dumps(row, sort_keys=True,
            separators=(",", ":")) + "\n").encode()
        # Hash mismatch is intentionally not hidden; refresh authority below.
        _resign(args)
    elif mutation == "remasked":
        args["legal_rows"][0]["pairs"][1]["status"] = "unavailable"
    elif mutation == "preempted":
        args["replay_rows"][0]["native_goal_preempted"] = True
        args["replay_bytes"] = (json.dumps(args["replay_rows"][0], sort_keys=True,
            separators=(",", ":")) + "\n").encode()
        _resign(args)
    elif mutation == "mismatch":
        args["replay_rows"][0]["behavior_id"] = 1
        args["replay_bytes"] = (json.dumps(args["replay_rows"][0], sort_keys=True,
            separators=(",", ":")) + "\n").encode()
        _resign(args)
    elif mutation == "duplicate":
        args["sidecar_bytes"] += args["sidecar_bytes"]
        _resign(args)
    elif mutation == "missing":
        args["sidecar_bytes"] = b""
        _resign(args)
    elif mutation == "heldout":
        args["split"] = "validation"
    with pytest.raises(ValueError, match=message):
        authority.validate_human_choice_source(**args)


def _resign(args):
    """Refresh all bound hashes using a key trusted by the fixture."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    # The fixture private key is not exposed; derive a deterministic test key.
    private = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    args["trusted_keys"] = {"reviewer-key": public}
    payload = dict(args["authority_manifest"]["payload"])
    payload.update(sidecar_sha256=authority.sha256_bytes(args["sidecar_bytes"]),
        source_sha256=authority.sha256_bytes(args["source_bytes"]),
        source_manifest_sha256=authority.sha256_bytes(args["source_manifest_bytes"]),
        replay_sha256=authority.sha256_bytes(args["replay_bytes"]),
        replay_manifest_sha256=authority.sha256_bytes(args["replay_manifest_bytes"]))
    signed = {"schema": authority.SCHEMA, "payload": payload,
        "signature_ed25519": private.sign(authority.canonical_bytes(payload)).hex()}
    args["authority_manifest"] = signed
