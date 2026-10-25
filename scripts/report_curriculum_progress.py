#!/usr/bin/env python3
"""Summarize saved curriculum evidence without loading a policy or simulator."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import html
import json
import math
import os
from pathlib import Path
import tempfile


def _read(path):
    return json.loads(path.read_text())


def _stats_files(root):
    """Yield stats files without following symlinked campaign directories."""
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
        if 'stats.json' in files:
            yield Path(current) / 'stats.json'


def _tree_has_symlink(root):
    for current, dirs, files in os.walk(root, followlinks=False):
        if any((Path(current) / name).is_symlink() for name in dirs + files):
            return True
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
    return False


def _queue_lane_directory(root, value):
    """Resolve a queue directory only when it is an existing in-root directory."""
    if not isinstance(value, str) or not value:
        raise ValueError('queue lane has no valid directory')
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('queue lane directory must be a safe relative path')
    lexical = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError('queue lane directory traverses a symlink')
    try:
        directory = lexical.resolve(strict=True)
    except OSError as exc:
        raise ValueError('queue lane directory does not exist') from exc
    if directory == root or root not in directory.parents:
        raise ValueError('queue lane directory is outside the report root')
    if not directory.is_dir():
        raise ValueError('queue lane directory is not a directory')
    return directory


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def category_summary(report, result, category, record):
    count = category['episodes']
    seeds = category.get('seeds', [])
    reference_seeds = category.get('reference_seeds', [])
    paired = bool(category.get('paired') is True and count > 0
                  and isinstance(report.get('protocol_id'), str) and report['protocol_id']
                  and report.get('sources')
                  and len(seeds) == count and len(set(seeds)) == count
                  and seeds == reference_seeds
                  and category.get('reference_protocol_id') == result.get('protocol_id')
                  and result.get('protocol_id') == report.get('protocol_id'))
    learner = category['useful_throughput']
    baseline = category['reference_throughput']
    valid = _finite(learner) and _finite(baseline) and learner >= 0 and baseline >= 0
    if not valid:
        raise ValueError('invalid or nonfinite throughput evidence')
    ratio = learner / baseline if paired and valid and baseline > 0 else None
    baseline_successes = record.get('reference_successes')
    source_fingerprint = _digest(report.get('sources', {}))
    key = {'protocol': report.get('protocol_id'), 'sources': source_fingerprint,
           'curriculum_version': report.get('curriculum_version'),
           'scenario_version': result.get('scenario_version'),
           'scenario': result['scenario_id'], 'category': category['category'],
           'split': report.get('split'), 'seeds': seeds, 'diagnostic': report.get('diagnostic', True)}
    return {'scenario': result['scenario_id'], 'category': category['category'],
            'evaluation_id': report.get('evaluation_id'), 'policy_id': report.get('policy_id'),
            'checkpoint_sha256': report.get('checkpoint_file_sha256'),
            'episodes': count, 'successes': category['successes'],
            'success_rate': category['successes'] / count if count else None,
            'success_interval_95': record.get('success_interval_95'),
            'reference_success_rate': baseline_successes / count if baseline_successes is not None and count else None,
            'reference_interval_95': record.get('reference_interval_95'),
            'useful_throughput': learner, 'reference_throughput': baseline,
            'throughput_ratio': ratio, 'paired': paired,
            'reference_qualified': category.get('reference_qualified') is True,
            'diagnostic': report.get('diagnostic', True),
            'curriculum_version': report.get('curriculum_version'),
            'scenario_version': result.get('scenario_version'),
            'source_fingerprint': source_fingerprint,
            'comparison_group': _digest(key), 'protocol_id': report.get('protocol_id'),
            'split': report.get('split'), 'seeds': seeds,
            'warnings': ([] if paired else ['unverified scenario/seed/protocol pairing'])
                        + ([] if valid else ['invalid throughput'])
                        + ([] if category.get('reference_qualified') else ['deterministic reference not qualified'])}


def _evidence_rows(evidence, sources, cache_dir=None):
    """Cache derived rows, never raw episode records or rejected evidence."""
    evidence = evidence.resolve()
    before = evidence.stat()
    key = {'version': 2, 'path': str(evidence), 'mtime_ns': before.st_mtime_ns,
           'size': before.st_size, 'stats_sources': _digest(sources)}
    cache = Path(cache_dir)/(_digest(key)+'.json') if cache_dir is not None else None
    if cache is not None:
        try:
            saved = _read(cache)
            rows = saved['rows']
            if (saved['key'] == key and isinstance(rows, list)
                    and all(isinstance(row, dict) for row in rows)
                    and saved['rows_sha256'] == _digest(rows)):
                return rows
        except (OSError, ValueError, KeyError, TypeError):
            pass
    report = _read(evidence)
    if report.get('sources') != sources:
        raise ValueError('evaluation source differs from run stats')
    number = int(evidence.stem.rsplit('-', 1)[-1])
    evaluation_results = report.get('evaluation_results', [])
    if not isinstance(evaluation_results, list):
        raise ValueError('evaluation results must be a list')
    scenario_ids = [result.get('scenario_id') for result in evaluation_results
                    if isinstance(result, dict)]
    if len(scenario_ids) != len(evaluation_results) or len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError('evaluation results are malformed or duplicated')
    records = {(r['scenario_id'], r['category']): r for r in report.get('records', [])}
    rows = []
    for result in evaluation_results:
        categories = result['categories']
        if not isinstance(categories, list):
            raise ValueError('evaluation categories must be a list')
        names = [category.get('category') for category in categories if isinstance(category, dict)]
        if len(names) != len(categories) or len(set(names)) != len(names):
            raise ValueError('evaluation categories are malformed or duplicated')
        for category in categories:
            row = category_summary(report, result, category,
                records.get((result['scenario_id'], category['category']), {}))
            rows.append(row | {'update': number, 'evidence': str(evidence)})
    after = evidence.stat()
    if (after.st_mtime_ns, after.st_size) != (before.st_mtime_ns, before.st_size):
        raise ValueError('evaluation changed while being read; excluded until next report')
    if cache is not None:
        try:
            _write(cache, json.dumps({'key': key, 'rows': rows, 'rows_sha256': _digest(rows)},
                                     allow_nan=False))
        except OSError:
            pass  # A read-only/full cache must not exclude valid simulation evidence.
    return rows


def summarize(root, cache_dir=None):
    root = Path(root)
    if any(part.is_symlink() for part in (root.absolute(), *root.absolute().parents)):
        raise ValueError('curriculum root traverses a symlink')
    root = root.resolve()
    runs, errors = [], []
    tree_has_symlink = _tree_has_symlink(root)
    if tree_has_symlink:
        errors.append({'path': '.', 'error': 'symlinked campaign evidence was excluded'})
    # Each stats file identifies a distinct training run. Evaluation files alone
    # never imply a queued or active learner.
    for path in sorted(_stats_files(root)):
        try:
            if path.is_symlink() or not path.is_file():
                errors.append({'path': str(path.relative_to(root)), 'error': 'stats file is not a regular in-root file'})
                continue
            stats = _read(path)
            if 'curriculum_state' not in stats:
                continue
            run = path.parent
            state = stats['curriculum_state']
            rows = stats.get('stats', [])
            evaluations = []
            evaluation_history_incomplete = tree_has_symlink
            for evidence in run.glob('evaluation-*.json'):
                if evidence.is_symlink() or not evidence.is_file():
                    evaluation_history_incomplete = True
                    errors.append({'path': str(evidence.relative_to(root)), 'error': 'evaluation is not a regular in-root file'})
                    continue
                try:
                    evaluations.extend(_evidence_rows(evidence, stats.get('sources'), cache_dir))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    evaluation_history_incomplete = True
                    errors.append({'path': str(evidence), 'error': str(exc)})
            latest = {}
            for row in sorted(evaluations, key=lambda r: r['update']):
                latest[row['scenario'], row['category']] = row
            qualification_evidence = []
            state = stats['curriculum_state']
            for scenario in state.get('qualified', []):
                checkpoint = state.get('last_qualified_checkpoint', {}).get(scenario)
                protocol = state.get('qualified_protocol', {}).get(scenario)
                matching = [row for row in evaluations
                            if row['scenario'] == scenario and row['policy_id'] == checkpoint
                            and row['protocol_id'] == protocol and row['split'] == 'validation']
                if matching:
                    evaluation_id = max(matching, key=lambda row: row['update'])['evaluation_id']
                    qualification_evidence.append({'scenario': scenario,
                        'evaluation_id': evaluation_id, 'policy_id': checkpoint,
                        'protocol_id': protocol,
                        'rows': [row for row in matching if row['evaluation_id'] == evaluation_id]})
            config = stats.get('config', {})
            configured_eval_every = config.get('eval_every') if isinstance(config, dict) else None
            eval_every = (configured_eval_every if type(configured_eval_every) is int
                          and configured_eval_every > 0 else None)
            runs.append({'run': str(run.relative_to(root)), 'path': str(run),
                         'algorithm': stats.get('algorithm', config.get('algorithm', 'ppo')),
                         'architecture': stats.get('architecture', config.get('architecture', run.name)),
                         'updates': stats.get('updates', rows[-1]['update'] if rows else 0),
                         # TrainingConfig is saved under stats.config. Invalid or
                         # missing cadence stays unavailable; never infer it.
                         'eval_every': eval_every,
                         'current': stats.get('current', rows[-1].get('scenario_id') if rows else None),
                         'qualified': state.get('qualified', []),
                         'consecutive_passes': state.get('consecutive_passes', {}),
                         'last_qualified_checkpoint': state.get('last_qualified_checkpoint', {}),
                         'qualified_protocol': state.get('qualified_protocol', {}),
                         'consecutive_protocol': state.get('consecutive_protocol', {}),
                         'evaluation_history_incomplete': evaluation_history_incomplete,
                         'last_promotion': next((r['promotion'] for r in reversed(rows) if 'promotion' in r), None),
                         'diagnostic': stats.get('diagnostic', True),
                         'latest_evaluations': list(latest.values()),
                         'evaluation_history': evaluations,
                         'qualification_evidence': qualification_evidence})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({'path': str(path), 'error': str(exc)})
    groups = {}
    for run in runs:
        for row in run['latest_evaluations']:
            groups.setdefault(row['comparison_group'], []).append(
                {'run': run['run'], 'algorithm': run.get('algorithm', 'ppo'),
                 'architecture': run['architecture'], **row})
    queue = None
    queue_lanes = []
    for name in ('orchestration.json', 'queue-status.json', 'queue.json', 'manifest.json'):
        candidate = root / name
        if candidate.exists() or candidate.is_symlink():
            try:
                if candidate.is_symlink() or not candidate.is_file():
                    raise ValueError('queue record is not a regular in-root file')
                queue = {'path': str(candidate), 'data': _read(candidate),
                         'age_seconds': max(0., datetime.now(timezone.utc).timestamp() - candidate.stat().st_mtime)}
            except (OSError, ValueError) as exc:
                errors.append({'path': str(candidate), 'error': str(exc)})
            break
    if queue and isinstance(queue['data'], dict):
        by_path = {Path(r['path']).resolve(): r for r in runs}
        lanes = queue['data'].get('lanes', [])
        if not isinstance(lanes, list):
            errors.append({'path': str(queue['path']), 'error': 'queue lanes must be a list'})
            lanes = []
        for index, lane in enumerate(lanes):
            try:
                if not isinstance(lane, dict):
                    raise ValueError('queue lane is malformed')
                directory = None
                if 'run_directory' in lane:
                    run_directory = lane['run_directory']
                    track = lane.get('track')
                    path = Path(run_directory) if isinstance(run_directory, str) else Path()
                    if (not isinstance(run_directory, str) or path.is_absolute()
                            or len(path.parts) != 1 or path.parts[0] in ('.', '..')
                            or not isinstance(track, str) or run_directory != track):
                        raise ValueError('queue run_directory must name its track directory')
                    directory = _queue_lane_directory(root, run_directory)
                if 'directory' in lane:
                    legacy_directory = _queue_lane_directory(root, lane['directory'])
                    if directory is not None and legacy_directory != directory:
                        raise ValueError('queue directory conflicts with run_directory')
                    directory = legacy_directory
                if directory is None:
                    raise ValueError('queue lane has no valid directory')
                run = by_path.get(directory)
                if run is None:
                    run = {'run': str(directory.relative_to(root)), 'path': str(directory),
                           'algorithm': lane.get('algorithm', 'ppo'),
                           'architecture': lane.get('architecture', 'unknown'),
                           'updates': lane.get('completed_updates', 0),
                           'current': lane.get('current'), 'qualified': lane.get('qualified', []),
                           'consecutive_passes': {}, 'last_promotion': None, 'diagnostic': False,
                           'latest_evaluations': []}
                    runs.append(run)
                run['queue_status'] = lane.get('status', 'unknown')
                run['device'] = lane.get('device')
                run['training_seed'] = lane.get('seed')
                queue_lanes.append({key: lane.get(key) for key in
                                    ('track', 'status', 'completed_updates', 'device', 'current', 'evaluation')})
                if lane.get('current'):
                    run['current'] = lane['current']
            except (KeyError, TypeError, ValueError, OSError) as exc:
                errors.append({'path': str(queue['path']), 'error': f'queue lane {index}: {exc}'})
    return {'generated_at': datetime.now(timezone.utc).isoformat(), 'root': str(root),
            'baseline': 'paired deterministic observed teacher, generic NN execution; different from deployed native deterministic controller',
            'runs': runs, 'comparison_groups': groups, 'queue': queue, 'queue_lanes': queue_lanes, 'errors': errors,
            'notes': ['Training reward is not a success metric.',
                      'Only rows sharing comparison_group have matching scenario, category, split, seeds, source and protocol.',
                      'Matching evaluation groups can have different training budgets; inspect update counts.',
                      'A ratio against an unqualified reference is descriptive and cannot qualify promotion.',
                      'No inference of process liveness is made from saved training stats.']}


def display_time(value):
    return datetime.fromisoformat(value).astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")


def markdown(report):
    lines = ['# Curriculum progress', '', f"Updated {display_time(report['generated_at'])}", '',
             'Baseline: paired deterministic observed teacher using generic NN execution.', '',
             '| Run | Algorithm | Queue status | Updates | Current/practice scenario | Qualified stages | Pass streaks |',
             '|---|---|---|---:|---|---:|---|']
    for run in report['runs']:
        streaks = ', '.join(f'{s}: {n}/3' for s, n in run['consecutive_passes'].items()) or 'none'
        lines.append(f"| {run['run']} | {run.get('algorithm', 'ppo')} | {run.get('queue_status', 'not tracked')} | {run['updates']} | {run['current'] or 'pending'} | {len(run['qualified'])} | {streaks} |")
    if not report['runs']:
        lines.append('| No completed training update yet | — | pending | 0 | pending | 0 | none |')
    lines += ['', 'Success and throughput are kept separate for every scenario/category. Ratios below an unqualified deterministic reference are descriptive only.', '',
              '| Run | Update | Scenario/category | Success | Deterministic success | Useful throughput / baseline | Ratio | Evidence |',
              '|---|---:|---|---:|---:|---|---:|---|']
    def pct(value):
        return f'{100*value:.1f}%' if value is not None else 'unavailable'
    for run in report['runs']:
        for row in run['latest_evaluations']:
            ratio = f"{row['throughput_ratio']:.3f}" if row['throughput_ratio'] is not None else 'unavailable'
            status = ', '.join((['diagnostic'] if row['diagnostic'] else []) + row['warnings']) or 'qualified paired reference'
            lines.append(f"| {run['run']} ({run.get('algorithm', 'ppo')}) | {row['update']} | {row['scenario']}/{row['category']} | {pct(row['success_rate'])} | {pct(row['reference_success_rate'])} | {row['useful_throughput']:.4g} / {row['reference_throughput']:.4g} | {ratio} | [{status}](<{row['evidence']}>) |")
    lines += ['', *report['notes']]
    if report['errors']:
        lines += ['', 'Evidence read errors (excluded from comparison):', '',
                  *[f"- {e['path']}: {e['error']}" for e in report['errors']]]
    return '\n'.join(lines) + '\n'


def webpage(report):
    """A static, self-refreshing evidence table; serving needs no simulator."""
    escape = lambda value: html.escape(str(value))
    def table(headers, rows):
        return '<table><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
            '<tr>' + ''.join(f'<td>{escape(v)}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table>'
    queue = report.get('queue')
    manifest = queue['data'] if queue else {}
    age = queue.get('age_seconds', 0) if queue else None
    counts = {}
    for run in report['runs']:
        status = run.get('queue_status', 'not tracked')
        counts[status] = counts.get(status, 0) + 1
    architectures = {r['architecture'] for r in report['runs']}
    algorithms = {r.get('algorithm', 'ppo') for r in report['runs']}
    overview = f"{len(architectures)} architectures / {len(algorithms)} algorithms / {len(report['runs'])} training lanes. " + ', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))
    state = f"Queue: {manifest.get('status', 'unavailable')}. "
    state += f'Manifest last written {age:.0f} seconds ago.' if age is not None else 'No queue manifest available.'
    if age is not None and age > 180:
        state += ' HEARTBEAT STALE: inspect the queue service before trusting running status.'
    source = manifest.get('source_identity_sha256', 'unavailable')
    rows = []
    for run in report['runs']:
        streaks = ', '.join(f'{s}: {n}/3' for s, n in run['consecutive_passes'].items()) or 'none'
        rows.append((run['run'], run.get('algorithm', 'ppo'), run.get('queue_status', 'not tracked'), run['updates'], run['current'] or 'pending',
                     len(run['qualified']), streaks))
    metrics = []
    for run in report['runs']:
        for row in run['latest_evaluations']:
            percent = lambda x: f'{x*100:.1f}%' if x is not None else 'unavailable'
            notes = (['diagnostic'] if row['diagnostic'] else []) + row['warnings']
            metrics.append((run['run'] + ' (' + run.get('algorithm', 'ppo') + ')', row['update'], row['scenario'] + '/' + row['category'],
                            percent(row['success_rate']), percent(row['reference_success_rate']),
                            f"{row['useful_throughput']:.4g} / {row['reference_throughput']:.4g}",
                            f"{row['throughput_ratio']:.3f}" if row['throughput_ratio'] is not None else 'unavailable',
                            '; '.join(notes) or 'qualified paired reference'))
    errors = ''.join(f"<li>{escape(e['path'])}: {escape(e['error'])}</li>" for e in report['errors'])
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<meta http-equiv="refresh" content="30"><title>Curriculum training progress</title>'
            '<style>body{font:16px system-ui,sans-serif;margin:2rem;color:#17202a}table{border-collapse:collapse;display:block;overflow:auto;margin:1rem 0;width:100%}'
            'th,td{text-align:left;vertical-align:top;padding:.55rem;border:1px solid #abb2b9}th{background:#edf0f2}p{max-width:100ch}code{overflow-wrap:anywhere}</style>'
            '<h1>Curriculum training progress</h1>'
            f'<p>{escape(overview)}</p><p>{escape(state)}</p><p>Report generated {escape(display_time(report["generated_at"]))}; refreshes every 30 seconds.</p>'
            f'<p>Frozen source identity: <code>{escape(source)}</code></p>'
            '<p>Baseline: paired deterministic observed teacher using generic NN execution. This is a scripted reference under the learner’s execution constraints; it differs from the deployed native deterministic controller.</p>'
            '<p>Each model starts from scratch and advances through physical-outcome gates. Unqualified reference ratios are descriptive; reward alone does not establish success.</p>'
            '<p><a href="progress.json">Machine-readable evidence</a> · <a href="progress.md">Text report</a></p>'
            + table(('Run', 'Algorithm', 'Queue status', 'Updates', 'Current/practice scenario', 'Qualified stages', 'Pass streaks'), rows)
            + '<h2>Latest paired evaluations</h2><p>No rows means evaluation has not completed. Compare architectures only when scenario, category, seeds and protocol match; see comparison_group in JSON. Different update counts reflect different training budgets.</p>'
            + table(('Run', 'Update', 'Scenario/category', 'Success', 'Deterministic success', 'Useful throughput / baseline', 'Ratio', 'Qualification'), metrics)
            + ('<h2>Evidence errors</h2><ul>' + errors + '</ul>' if errors else '') + '</html>\n')


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.root / 'progress.json'
    report = summarize(args.root, cache_dir=output.parent / '.evidence-cache')
    _write(output, json.dumps(report, indent=2, allow_nan=False) + '\n')
    _write(output.with_suffix('.md'), markdown(report))
    _write(output.parent / 'performance.html', webpage(report))
    print(json.dumps({'output': str(output), 'runs': len(report['runs']), 'errors': len(report['errors'])}))


if __name__ == '__main__':
    main()
