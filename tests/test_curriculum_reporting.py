"""Saved-evidence reporting tests; no simulated behavior is asserted here."""
import importlib.util
import json
import os
from pathlib import Path


spec = importlib.util.spec_from_file_location('curriculum_report', Path(__file__).resolve().parents[1] / 'scripts/report_curriculum_progress.py')
reporting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting)


def evidence(protocol='suite', seeds=(1, 5), reference_seeds=(1, 5), qualified=True):
    category = {'category': 'cluster', 'episodes': 2, 'successes': 1,
                'useful_throughput': .5, 'reference_throughput': 1., 'paired': True,
                'reference_qualified': qualified, 'reference_protocol_id': protocol,
                'seeds': list(seeds), 'reference_seeds': list(reference_seeds)}
    result = {'scenario_id': 'pickup_simple', 'scenario_version': '1', 'protocol_id': protocol,
              'categories': [category]}
    report = {'protocol_id': protocol, 'sources': {'sim': 'digest'}, 'curriculum_version': '1',
              'evaluation_id': 'round', 'policy_id': 'policy', 'split': 'validation',
              'diagnostic': True, 'evaluation_results': [result],
              'records': [{'scenario_id': 'pickup_simple', 'category': 'cluster', 'reference_successes': 2}]}
    return report, result, category


def test_pairing_required_before_relative_throughput():
    report, result, category = evidence(reference_seeds=(9, 13))
    row = reporting.category_summary(report, result, category, {})
    assert not row['paired'] and row['throughput_ratio'] is None
    category['reference_seeds'] = category['seeds']
    category['reference_protocol_id'] = 'another-device'
    assert reporting.category_summary(report, result, category, {})['throughput_ratio'] is None


def test_reference_qualification_and_comparison_groups_remain_distinct():
    report, result, category = evidence(qualified=False)
    row = reporting.category_summary(report, result, category, {})
    assert row['throughput_ratio'] == .5 and not row['reference_qualified']
    assert 'deterministic reference not qualified' in row['warnings']
    report2, result2, category2 = evidence(seeds=(9, 13), reference_seeds=(9, 13))
    row2 = reporting.category_summary(report2, result2, category2, {})
    assert row['comparison_group'] != row2['comparison_group']


def test_queue_includes_not_started_variants_and_latest_saved_round(tmp_path):
    run = tmp_path / 'variants/full_field_cnn/seed-7'
    run.mkdir(parents=True)
    report, _, _ = evidence()
    (run / 'stats.json').write_text(json.dumps({'sources': report['sources'], 'updates': 10,
        'algorithm': 'sac', 'architecture': 'full_field_cnn', 'curriculum_state': {'qualified': [], 'consecutive_passes': {'pickup_simple': 1}},
        'stats': [{'update': 10, 'scenario_id': 'pickup_simple'}]}))
    (run / 'evaluation-10.json').write_text(json.dumps(report))
    report['evaluation_id'] = 'older'
    (run / 'evaluation-2.json').write_text(json.dumps(report))
    lanes = [{'algorithm': 'sac', 'architecture': 'full_field_cnn', 'seed': 7, 'directory': str(run), 'status': 'running', 'current': 'pickup_simple'},
             {'architecture': 'field_fno', 'seed': 7, 'directory': 'variants/field_fno/seed-7', 'status': 'queued'}]
    (tmp_path / 'orchestration.json').write_text(json.dumps({'lanes': lanes}))
    summary = reporting.summarize(tmp_path)
    assert not summary['errors'] and len(summary['runs']) == 2
    cnn = next(r for r in summary['runs'] if r['architecture'] == 'full_field_cnn')
    assert cnn['queue_status'] == 'running' and cnn['latest_evaluations'][0]['update'] == 10
    assert cnn['algorithm'] == 'sac'
    assert cnn['latest_evaluations'][0]['reference_success_rate'] == 1
    assert 'field_fno' in reporting.markdown(summary)
    assert '| Algorithm |' in reporting.markdown(summary)
    assert '(sac)' in reporting.markdown(summary)


def test_source_mismatch_and_partial_evidence_are_reported_not_ranked(tmp_path):
    (tmp_path / 'stats.json').write_text(json.dumps({'sources': {'sim': 'new'}, 'curriculum_state': {}, 'stats': []}))
    report, _, _ = evidence()
    (tmp_path / 'evaluation-1.json').write_text(json.dumps(report))
    (tmp_path / 'evaluation-2.json').write_text('{')
    summary = reporting.summarize(tmp_path)
    assert len(summary['errors']) == 2
    assert summary['comparison_groups'] == {}


def test_static_page_identifies_stale_queue_and_reference_execution():
    summary = {'generated_at': '2026-10-21T00:00:00Z', 'runs': [], 'errors': [],
               'queue': {'age_seconds': 181, 'data': {'status': 'running', 'source_identity_sha256': 'frozen'}}}
    page = reporting.webpage(summary)
    assert 'HEARTBEAT STALE' in page
    assert 'content="30"' in page
    assert 'differs from the deployed native deterministic controller' in page


def test_missing_protocol_cannot_manufacture_pairing():
    report, result, category = evidence(protocol=None)
    assert not reporting.category_summary(report, result, category, {})['paired']


def test_evidence_cache_avoids_reparse_and_invalidates_source_and_file_changes(tmp_path, monkeypatch):
    report, _, _ = evidence()
    stats_path = tmp_path/'stats.json'
    stats = {'sources': report['sources'], 'curriculum_state': {}, 'stats': []}
    stats_path.write_text(json.dumps(stats))
    saved = tmp_path/'evaluation-1.json'
    saved.write_text(json.dumps(report))
    cache = tmp_path/'report/.evidence-cache'
    original_read = reporting._read
    reads = []
    def read(path):
        if Path(path) == saved:
            reads.append(path)
        return original_read(path)
    monkeypatch.setattr(reporting, '_read', read)
    first = reporting.summarize(tmp_path, cache_dir=cache)
    second = reporting.summarize(tmp_path, cache_dir=cache)
    assert len(reads) == 1 and first['comparison_groups'] == second['comparison_groups']
    assert len(list(cache.glob('*.json'))) == 1
    stamp = saved.stat()
    os.utime(saved, ns=(stamp.st_atime_ns, stamp.st_mtime_ns+1))
    reporting.summarize(tmp_path, cache_dir=cache)
    assert len(reads) == 2  # mtime invalidates even when bytes/size are unchanged
    stats['sources'] = {'sim': 'changed'}
    stats_path.write_text(json.dumps(stats))
    rejected = reporting.summarize(tmp_path, cache_dir=cache)
    assert len(reads) == 3 and len(rejected['errors']) == 1
    assert rejected['comparison_groups'] == {}
    assert len(list(cache.glob('*.json'))) == 2  # rejected source never cached


def test_partial_evidence_and_corrupt_cache_are_not_reused(tmp_path):
    report, _, _ = evidence()
    (tmp_path/'stats.json').write_text(json.dumps({'sources': report['sources'], 'curriculum_state': {}, 'stats': []}))
    saved = tmp_path/'evaluation-1.json'
    saved.write_text('{')
    cache = tmp_path/'cache'
    assert reporting.summarize(tmp_path, cache_dir=cache)['errors']
    assert not list(cache.glob('*.json'))
    saved.write_text(json.dumps(report))
    expected = reporting.summarize(tmp_path, cache_dir=cache)
    cached = next(cache.glob('*.json'))
    cached.write_text('{')
    rebuilt = reporting.summarize(tmp_path, cache_dir=cache)
    assert rebuilt['comparison_groups'] == expected['comparison_groups']
    assert not rebuilt['errors']
    assert json.loads(cached.read_text())['rows']
