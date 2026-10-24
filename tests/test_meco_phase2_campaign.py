import hashlib
import json
from pathlib import Path

import pytest

from frc_defense import meco_phase2_campaign as campaign


def _protocol(path):
    body = {"protocol_version": "meco-qualification-v1", "cases": [
        {"scenario_id": "pickup_simple", "category": "single", "split": "train", "seeds": [11, 12]},
        {"scenario_id": "pickup_simple", "category": "single", "split": "validation", "seeds": [21]},
        {"scenario_id": "pickup_simple", "category": "single", "split": "test", "seeds": [31]},
    ]}
    body["protocol_id"] = hashlib.sha256(campaign._canonical(body).encode()).hexdigest()
    path.write_text(json.dumps(body))
    return path


@pytest.fixture
def setup_case(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "frc_defense").mkdir()
    protocol = _protocol(tmp_path / "protocol.json")
    monkeypatch.setattr(campaign, "_source_snapshot", lambda _root: ("a" * 40, "b" * 64))
    root = tmp_path / "outputs" / "campaign"
    value = campaign.describe_campaign(protocol_path=protocol,
        scenario_id="pickup_simple", category="single", seeds=[11, 12],
        device="cuda:0", output_root=root, repo_root=repo)
    return repo, protocol, root, value


def test_description_is_no_write_no_lease_and_binds_fixed_c1_c2(setup_case):
    _, _, root, descriptor = setup_case
    assert not root.exists()
    assert descriptor["tracks"] == ["C1", "C2"]
    assert descriptor["split"] == "train"
    assert {key: descriptor[key] for key in campaign.FIXED} == campaign.FIXED
    assert descriptor["phase2_ready"] is False
    assert descriptor["qualification_eligible"] is False
    campaign.validate_descriptor(descriptor, repo_root=Path(descriptor["qualification_protocol"]["path"]).parent / "repo")
    assert not root.exists()


@pytest.mark.parametrize("seeds", [[21], [11, 11], [11, 99], []])
def test_rejects_heldout_duplicate_unknown_and_empty_seeds(setup_case, seeds):
    repo, protocol, root, _ = setup_case
    with pytest.raises(ValueError, match="seeds"):
        campaign.describe_campaign(protocol_path=protocol, scenario_id="pickup_simple",
            category="single", seeds=seeds, device="cuda:0", output_root=root, repo_root=repo)


def test_rejects_existing_root_symlink_and_overlapping_paths(setup_case, tmp_path):
    repo, protocol, root, _ = setup_case
    root.parent.mkdir()
    root.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        campaign.describe_campaign(protocol_path=protocol, scenario_id="pickup_simple",
            category="single", seeds=[11], device="cuda:0", output_root=root, repo_root=repo)
    root.rmdir()
    target = tmp_path / "real-output"
    target.mkdir()
    link = tmp_path / "link-output"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        campaign.describe_campaign(protocol_path=protocol, scenario_id="pickup_simple",
            category="single", seeds=[11], device="cuda:0", output_root=link / "run", repo_root=repo)
    with pytest.raises(ValueError, match="overlap"):
        campaign.describe_campaign(protocol_path=protocol, scenario_id="pickup_simple",
            category="single", seeds=[11], device="cuda:0", output_root=repo / "out", repo_root=repo)


def test_rejects_protocol_symlink_and_tampered_descriptor(setup_case, tmp_path):
    repo, protocol, root, descriptor = setup_case
    link = tmp_path / "protocol-link.json"
    link.symlink_to(protocol)
    with pytest.raises(ValueError, match="symlink"):
        campaign.describe_campaign(protocol_path=link, scenario_id="pickup_simple",
            category="single", seeds=[11], device="cuda:0", output_root=root, repo_root=repo)
    changed = dict(descriptor, device="cuda:1")
    with pytest.raises(ValueError, match="digest"):
        campaign.validate_descriptor(changed, repo_root=repo)


def test_validation_rejects_stale_source_and_changed_protocol(setup_case, monkeypatch):
    repo, protocol, _, descriptor = setup_case
    monkeypatch.setattr(campaign, "_source_snapshot", lambda _root: ("c" * 40, "d" * 64))
    with pytest.raises(ValueError, match="source identity is stale"):
        campaign.validate_descriptor(descriptor, repo_root=repo)
    monkeypatch.setattr(campaign, "_source_snapshot", lambda _root: ("a" * 40, "b" * 64))
    content = json.loads(protocol.read_text())
    content["cases"][0]["seeds"] = [11, 13]
    protocol.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="ID does not match"):
        campaign.validate_descriptor(descriptor, repo_root=repo)


def test_write_descriptor_is_explicit_no_clobber(setup_case, tmp_path):
    repo, _, root, descriptor = setup_case
    output = tmp_path / "descriptor.json"
    campaign.write_descriptor(descriptor, output)
    assert output.is_file() and not root.exists()
    with pytest.raises(ValueError, match="already exists"):
        campaign.write_descriptor(descriptor, output)
    with pytest.raises(ValueError, match="must not overlap"):
        campaign.write_descriptor(descriptor, root / "descriptor.json")


class _Lease:
    def __init__(self, events):
        self.events = events
    def __enter__(self):
        self.events.append("lease-enter")
    def __exit__(self, *args):
        self.events.append("lease-exit")


def test_explicit_execution_collects_once_fits_c1_then_c2_validates_and_releases(setup_case, monkeypatch):
    repo, _, root, descriptor = setup_case
    events = []
    monkeypatch.setattr(campaign, "_repo_root", lambda: repo)
    sources = {"source": "test-runtime"}
    monkeypatch.setattr(campaign, "_runtime_source_identity", lambda: sources)
    def collect(path, **kwargs):
        events.append("collect")
        path.write_text('{"record":1}\n')
        manifest = {"dataset_sha256": campaign._sha(path.read_bytes()), "record_count": 1,
            "sources": sources, "source_identity_sha256": campaign._sha(campaign._canonical(sources).encode()),
            "protocol_id": descriptor["qualification_protocol"]["id"],
            "qualification_protocol_sha256": descriptor["qualification_protocol"]["sha256"],
            "split": "train", "scenario_id": "pickup_simple", "category": "single",
            "seeds": [11, 12], "candidate_k": 16, "prediction_mode": "cv",
            "horizon_physics_ticks": 160, "max_macro_ticks": 12,
            "max_plan_c_descriptors_per_prefix": 8, "max_plan_a_pairs_per_prefix": 8,
            "gamma_per_macro": 0.993}
        path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(manifest))
        return {"record_count": 1, "dataset_sha256": campaign._sha(path.read_bytes())}
    def fit(track, dataset, output, **kwargs):
        events.append("fit-" + track)
        output.write_bytes(track.encode())
    def validate(track, output, **kwargs):
        events.append("validate-" + track)
        return {"dataset_sha256": campaign._sha(Path(descriptor["outputs"]["dataset"]).read_bytes()),
            "non_tied_pair_count": 2, "updates": 10}
    result = campaign.execute_campaign(descriptor,
        lease_factory=lambda *_args, **_kwargs: _Lease(events),
        collect_fn=collect, fit_fn=fit, validate_fn=validate)
    assert events == ["lease-enter", "collect", "fit-C1", "validate-C1",
        "fit-C2", "validate-C2", "lease-exit"]
    assert result["status"] == "artifacts_validated"
    assert result["phase2_ready"] is False and result["qualification_eligible"] is False
    assert [row.get("track") for row in result["stages"] if row["stage"] == "fit_and_validate"] == ["C1", "C2"]


def test_failed_fit_records_failure_and_releases_lease(setup_case, monkeypatch):
    repo, _, root, descriptor = setup_case
    monkeypatch.setattr(campaign, "_repo_root", lambda: repo)
    sources = {"source": "test-runtime"}
    monkeypatch.setattr(campaign, "_runtime_source_identity", lambda: sources)
    events = []
    def collect(path, **kwargs):
        path.write_text('{"record":1}\n')
        manifest = {"dataset_sha256": campaign._sha(path.read_bytes()), "record_count": 1,
            "sources": sources, "source_identity_sha256": campaign._sha(campaign._canonical(sources).encode()),
            "protocol_id": descriptor["qualification_protocol"]["id"],
            "qualification_protocol_sha256": descriptor["qualification_protocol"]["sha256"],
            "split": "train", "scenario_id": "pickup_simple", "category": "single",
            "seeds": [11, 12], "candidate_k": 16, "prediction_mode": "cv",
            "horizon_physics_ticks": 160, "max_macro_ticks": 12,
            "max_plan_c_descriptors_per_prefix": 8, "max_plan_a_pairs_per_prefix": 8,
            "gamma_per_macro": 0.993}
        path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(manifest))
        return {"record_count": 1, "dataset_sha256": campaign._sha(path.read_bytes())}
    def fit(track, dataset, output, **kwargs):
        events.append(track)
        raise RuntimeError("synthetic fit failure")
    with pytest.raises(RuntimeError, match="synthetic fit failure"):
        campaign.execute_campaign(descriptor,
            lease_factory=lambda *_args, **_kwargs: _Lease(events), collect_fn=collect,
            fit_fn=fit, validate_fn=lambda *_args, **_kwargs: {})
    saved = json.loads((root / "campaign-report.json").read_text())
    assert saved["status"] == "failed"
    assert saved["phase2_ready"] is False and saved["qualification_eligible"] is False
    assert events == ["lease-enter", "C1", "lease-exit"]


def test_execution_requires_explicit_api_and_refuses_existing_root(setup_case):
    _, _, root, descriptor = setup_case
    root.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        campaign.execute_campaign(descriptor)
