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
    key = {'protocol': report.get('protocol_id'), 'sources': _digest(report.get('sources', {})),
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
            'comparison_group': _digest(key), 'protocol_id': report.get('protocol_id'),
            'split': report.get('split'), 'seeds': seeds,
            'warnings': ([] if paired else ['unverified scenario/seed/protocol pairing'])
                        + ([] if valid else ['invalid throughput'])
                        + ([] if category.get('reference_qualified') else ['deterministic reference not qualified'])}


def _evidence_rows(evidence, sources, cache_dir=None):
    """Cache derived rows, never raw episode records or rejected evidence."""
    evidence = evidence.resolve()
    before = evidence.stat()
    key = {'version': 1, 'path': str(evidence), 'mtime_ns': before.st_mtime_ns,
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
    records = {(r['scenario_id'], r['category']): r for r in report.get('records', [])}
    rows = []
    for result in report.get('evaluation_results', []):
        for category in result['categories']:
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
    root = Path(root).resolve()
    runs, errors = [], []
    # Each stats file identifies a distinct training run. Evaluation files alone
    # never imply a queued or active learner.
    for path in sorted(root.rglob('stats.json')):
        try:
            stats = _read(path)
            if 'curriculum_state' not in stats:
                continue
            run = path.parent
            state = stats['curriculum_state']
            rows = stats.get('stats', [])
            evaluations = []
            for evidence in run.glob('evaluation-*.json'):
                try:
                    evaluations.extend(_evidence_rows(evidence, stats.get('sources'), cache_dir))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append({'path': str(evidence), 'error': str(exc)})
            latest = {}
            for row in sorted(evaluations, key=lambda r: r['update']):
                latest[row['scenario'], row['category']] = row
            config = stats.get('config', {})
            runs.append({'run': str(run.relative_to(root)), 'path': str(run),
                         'algorithm': stats.get('algorithm', config.get('algorithm', 'ppo')),
                         'architecture': stats.get('architecture', config.get('architecture', run.name)),
                         'updates': stats.get('updates', rows[-1]['update'] if rows else 0),
                         'current': stats.get('current', rows[-1].get('scenario_id') if rows else None),
                         'qualified': state.get('qualified', []),
                         'consecutive_passes': state.get('consecutive_passes', {}),
                         'last_promotion': next((r['promotion'] for r in reversed(rows) if 'promotion' in r), None),
                         'diagnostic': stats.get('diagnostic', True),
                         'latest_evaluations': list(latest.values())})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({'path': str(path), 'error': str(exc)})
    groups = {}
    for run in runs:
        for row in run['latest_evaluations']:
            groups.setdefault(row['comparison_group'], []).append(
                {'run': run['run'], 'algorithm': run.get('algorithm', 'ppo'),
                 'architecture': run['architecture'], **row})
    queue = None
    for name in ('orchestration.json', 'queue-status.json', 'queue.json', 'manifest.json'):
        candidate = root / name
        if candidate.exists():
            try:
                queue = {'path': str(candidate), 'data': _read(candidate),
                         'age_seconds': max(0., datetime.now(timezone.utc).timestamp() - candidate.stat().st_mtime)}
            except (OSError, ValueError) as exc:
                errors.append({'path': str(candidate), 'error': str(exc)})
            break
    if queue and isinstance(queue['data'], dict):
        by_path = {Path(r['path']).resolve(): r for r in runs}
        for lane in queue['data'].get('lanes', []):
            directory = Path(lane['directory'])
            directory = directory.resolve() if directory.is_absolute() else (root / directory).resolve()
            run = by_path.get(directory)
            if run is None:
                run = {'run': str(directory.relative_to(root)), 'path': str(directory),
                       'algorithm': lane.get('algorithm', 'ppo'),
                       'architecture': lane['architecture'], 'updates': lane.get('completed_updates', 0),
                       'current': lane.get('current'), 'qualified': lane.get('qualified', []),
                       'consecutive_passes': {}, 'last_promotion': None, 'diagnostic': False,
                       'latest_evaluations': []}
                runs.append(run)
            run['queue_status'] = lane.get('status')
            run['device'] = lane.get('device')
            run['training_seed'] = lane.get('seed')
            if lane.get('current'):
                run['current'] = lane['current']
    return {'generated_at': datetime.now(timezone.utc).isoformat(), 'root': str(root),
            'baseline': 'paired deterministic observed teacher, generic NN execution; different from deployed native deterministic controller',
            'runs': runs, 'comparison_groups': groups, 'queue': queue, 'errors': errors,
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
