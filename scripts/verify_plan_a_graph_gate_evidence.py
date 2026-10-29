#!/usr/bin/env python3
"""Check whether closed-gate Plan A reports match a proposed graph-only gate.

This verifier does not enable dispatch, validate a live training run, or attest
that report files are authentic. Reviewers must retain and authenticate the
reports separately. Only the graph gate literal may differ from the candidate
source; all pinned code and fixture cases must remain identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from frc_defense.meco_plan_a_hip_parity_harness import (
    FIXTURE_RELATIVE, OUTPUT_KEYS, canonical_bytes, sha256_bytes, source_hashes,
)

GRAPH_SOURCE = "frc_defense/meco_plan_a_shortest_path.py"
CLOSED = b"PLAN_A_HIP_GRAPH_PARITY_PROFILE_GATE_PASSED = False"
OPEN = b"PLAN_A_HIP_GRAPH_PARITY_PROFILE_GATE_PASSED = True"
PREWEIGHTED_CLOSED = b"PLAN_A_HIP_PARITY_PROFILE_GATE_PASSED = False"


def git_bytes(root: Path, commit: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), "show", f"{commit}:{path}"],
        check=True, capture_output=True).stdout


def normalized_graph_source(source: bytes) -> bytes:
    if source.count(CLOSED) + source.count(OPEN) != 1 or source.count(PREWEIGHTED_CLOSED) != 1:
        raise ValueError("expected exactly one graph gate and a closed preweighted gate")
    return source.replace(OPEN, CLOSED)


def checked_paths(row: dict, report_path: Path) -> dict:
    paths = row.get("hip_paths")
    if not isinstance(paths, list) or any(not isinstance(item, dict) or
            not isinstance(item.get("path"), str) or not item["path"] for item in paths):
        raise ValueError(f"{report_path}: malformed HIP path records")
    by_path = {item["path"]: item for item in paths}
    if len(by_path) != len(paths):
        raise ValueError(f"{report_path}: duplicate HIP path records")
    return by_path


def check_report(root: Path, report_path: Path, expected_report_sha256: str,
                 current: dict[str, str], graph_source: bytes) -> dict:
    raw = report_path.read_bytes()
    if (len(expected_report_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in expected_report_sha256) or
            hashlib.sha256(raw).hexdigest() != expected_report_sha256):
        raise ValueError(f"{report_path}: report does not match its reviewed hash")
    report = json.loads(raw)
    if (report.get("schema_version") != 1 or
        report.get("evidence_class") != "hip_parity_profile_candidate" or
        report.get("passed") is not True or report.get("git_dirty") is not False or
        report.get("source_identity_verified_at_start_and_end") is not True or
        report.get("production_gates_open") != {"graph": False, "preweighted": False}):
        raise ValueError(f"{report_path}: report is not a clean closed-gate candidate")
    commit = report.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError(f"{report_path}: invalid candidate commit")
    pinned = report.get("source_sha256")
    if not isinstance(pinned, dict) or set(pinned) != set(current):
        raise ValueError(f"{report_path}: pinned source set differs")
    normalized = normalized_graph_source(graph_source)
    current = dict(current)
    current[GRAPH_SOURCE] = hashlib.sha256(normalized).hexdigest()
    if current != pinned:
        raise ValueError(f"{report_path}: source differs beyond the graph gate literal")
    for name, expected in pinned.items():
        if hashlib.sha256(git_bytes(root, commit, name)).hexdigest() != expected:
            raise ValueError(f"{report_path}: candidate commit does not bind {name}")
    old_manifest_bytes = git_bytes(root, commit, FIXTURE_RELATIVE)
    if hashlib.sha256(old_manifest_bytes).hexdigest() != report.get("fixture_manifest_sha256"):
        raise ValueError(f"{report_path}: candidate fixture identity mismatch")
    manifest = json.loads(old_manifest_bytes)
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
    if (manifest.get("manifest_payload_sha256") != sha256_bytes(canonical_bytes(unsigned)) or
            manifest.get("source_sha256") != pinned):
        raise ValueError(f"{report_path}: invalid candidate fixture manifest")
    present = json.loads((root / FIXTURE_RELATIVE).read_bytes())
    present_unsigned = {k: v for k, v in present.items() if k != "manifest_payload_sha256"}
    if (present.get("manifest_payload_sha256") != sha256_bytes(canonical_bytes(present_unsigned)) or
            present.get("source_sha256") != source_hashes(root)):
        raise ValueError(f"{report_path}: current fixture manifest is not source-bound")
    # The fixture's pinned source map and its canonical digest must change
    # when the graph gate changes; its case content must not.
    for key in set(manifest) | set(present):
        if key not in {"source_sha256", "manifest_payload_sha256"} and manifest.get(key) != present.get(key):
            raise ValueError(f"{report_path}: current fixture content differs in {key}")
    cases = manifest["cases"]
    rows = report.get("cases")
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise ValueError(f"{report_path}: incomplete case records")
    by_id = {row.get("id"): row for row in rows}
    if len(by_id) != len(cases) or set(by_id) != {case["id"] for case in cases}:
        raise ValueError(f"{report_path}: case identities differ")
    for case in cases:
        row = by_id[case["id"]]
        if row.get("graph_sha256") != case.get("graph_sha256") or row.get("tensor_sha256") != case.get("tensor_sha256"):
            raise ValueError(f"{report_path}: case input changed: {case['id']}")
        paths = checked_paths(row, report_path)
        is_graph = case.get("route_kind") == "geometry"
        if not is_graph:
            continue
        graph = paths.get("graph_geometry")
        if graph is None:
            raise ValueError(f"{report_path}: missing graph result: {case['id']}")
        if case.get("device_rejects"):
            if graph.get("status") != "candidate_rejected_input":
                raise ValueError(f"{report_path}: invalid graph was not rejected")
            continue
        comparisons = graph.get("compared_output_fields", [])
        elapsed = graph.get("hip_elapsed_ms")
        if (not isinstance(comparisons, list) or
                any(not isinstance(v, dict) for v in comparisons) or
                not isinstance(elapsed, list)):
            raise ValueError(f"{report_path}: graph parity/profile incomplete: {case['id']}")
        if (graph.get("status") != "candidate_pass" or graph.get("exact_output_match") is not True or
            graph.get("support_digest_match") is not True or
            graph.get("route_output_field_sha256") != case.get("route_output_field_sha256") or
            graph.get("support_digest_sha256") != case.get("support_digest_sha256") or
            {v.get("field") for v in comparisons if v.get("exact_equal") is True} != set(OUTPUT_KEYS) or
            len(comparisons) != len(OUTPUT_KEYS) or
            type(graph.get("warmups")) is not int or graph["warmups"] < 1 or
            type(graph.get("repeats")) is not int or graph["repeats"] < 3 or
            len(elapsed) != graph.get("repeats") or
            any(type(v) not in (int, float) or not 0 < v < float("inf") for v in elapsed)):
            raise ValueError(f"{report_path}: graph parity/profile incomplete: {case['id']}")
    identity = check_device(report, report_path)
    return {"report": str(report_path), "report_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_commit": commit, **identity}


def check_device(report: dict, report_path: Path) -> dict:
    device = report.get("device", {})
    requested = report.get("requested_device")
    auth = report.get("lease_authorization", {})
    attestation = auth.get("physical_attestation", {})
    devices = attestation.get("devices", [])
    if (requested not in ("cuda:0", "cuda:1") or device.get("index") != int(requested[-1]) or
        device.get("model") != "AMD Radeon Pro WX 9100" or
        auth.get("logical_lease_devices") != [requested] or not auth.get("logical_lease_id") or
        len(devices) != 1 or devices[0].get("device") != requested or
        not isinstance(device.get("uuid"), str) or not device["uuid"].strip() or
        devices[0].get("hip_uuid") != device["uuid"] or
        devices[0].get("model") != device["model"] or
        not isinstance(devices[0].get("pci_bdf"), str) or not devices[0]["pci_bdf"].strip()):
        raise ValueError(f"{report_path}: device or lease attestation incomplete")
    return {"device": requested, "uuid": device["uuid"], "pci_bdf": devices[0]["pci_bdf"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--report-sha256", action="append", required=True,
        help="Reviewed SHA-256 for the report in the corresponding --report position")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        current = source_hashes(root)
        graph_source = (root / GRAPH_SOURCE).read_bytes()
        if len(args.report) != len(args.report_sha256):
            raise ValueError("one reviewed SHA-256 is required for each report")
        reports = [check_report(root, path.resolve(), digest, current, graph_source)
                   for path, digest in zip(args.report, args.report_sha256)]
        if {r["device"] for r in reports} != {"cuda:0", "cuda:1"} or len(reports) != 2 or len({r["uuid"] for r in reports}) != 2 or len({r["pci_bdf"] for r in reports}) != 2:
            raise ValueError("exactly one report for each distinct WX9100 is required")
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"graph gate evidence rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"graph_evidence_matches": True, "production_gate_open": OPEN in graph_source,
        "reports": reports, "limitations": "Evidence matching only; independent review, final dispatch and GPU training validation remain required."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
