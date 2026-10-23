"""Audit the declared completed study and saved simulator outcome records."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import torch

MODELS = ('baseline_mlp', 'residual_mlp', 'deep_sets', 'entity_attention',
          'full_field_cnn', 'full_field_resnet', 'field_fno')
SCENARIOS = ('pickup_simple', 'pickup_scattered', 'deterministic_defense',
             'defense_3v3', 'offense_vs_deterministic_defense')


def audit(root: Path, expected_source: str) -> dict:
    expected = set(itertools.product(MODELS, SCENARIOS, (7, 19, 43), (.0003,)))
    expected.update(itertools.product(MODELS, ('deterministic_defense',), (7,), (.0001, .001)))
    found, errors, records = set(), [], []
    def check(condition, message):
        if not condition:
            errors.append(message)
    for path in sorted(root.rglob('metadata.json')):
        m = json.loads(path.read_text())
        key = (m['architecture'], m['scenario'], m['seed'], m['learning_rate'])
        label = '/'.join(map(str, key))
        check(key not in found, f'{label}: duplicate')
        found.add(key)
        check(m['status'] == 'complete', f'{label}: incomplete')
        if m['status'] != 'complete':
            continue
        check(m['provenance']['source_sha256'] == expected_source, f'{label}: source mismatch')
        check(m['perception_only_dynamic_inputs'], f'{label}: input provenance missing')
        check(m.get('perception_contract', {}).get('version') == 2,
              f'{label}: end-to-end perception contract missing')
        check(not m['runtime']['miopen_enabled'], f'{label}: backend mismatch')
        config, directory = m['config'], path.parent
        n_agents = 3 if m['scenario'] == 'defense_3v3' else 1
        horizon = config['horizon']
        check(config['envs'] == 8 and config['eval_envs'] == 4, f'{label}: world count mismatch')
        check(config['updates'] == 4 and config['action_repeat'] == 10, f'{label}: update/action cadence mismatch')
        check(config['rollout_steps'] * 10 == horizon == config['eval_ticks'], f'{label}: horizon mismatch')
        curve = json.loads((directory / 'learning_curve.json').read_text())
        check([row['update'] for row in curve] == [0, 2, 4], f'{label}: checkpoint coverage')
        last_weights = None
        changed = False
        for row in curve:
            u = row['update']
            record = json.loads((directory / f'eval-{u:04d}.json').read_text())
            ckpt = directory / f'policy-{u:04d}.pt'
            check(hashlib.sha256(ckpt.read_bytes()).hexdigest() == row['checkpoint_sha256'], f'{label}: checkpoint hash {u}')
            state = torch.load(ckpt, map_location='cpu', weights_only=True)['model_state_dict']
            check(all(torch.isfinite(tensor).all().item() for tensor in state.values()), f'{label}: nonfinite weights {u}')
            if last_weights is not None:
                changed |= any(not torch.equal(last_weights[k], value) for k, value in state.items())
            last_weights = state
            check(row['transitions'] == u * config['rollout_steps'] * 8 * n_agents, f'{label}: transition budget {u}')
            check(row['world_physics_ticks'] == u * horizon * 8, f'{label}: physics budget {u}')
            check(record['source_sha256'] == expected_source, f'{label}: eval source {u}')
            check(record['seed'] == m['seed'] + 100000, f'{label}: held-out seed {u}')
            check(record['physics_dt'] == .02 and record['ticks'] == horizon, f'{label}: real-time cadence {u}')
            check(record['perception']['enabled'], f'{label}: perception off {u}')
            check(record['frames'][0]['tick'] == 0 and record['frames'][-1]['tick'] == horizon, f'{label}: trajectory missing {u}')
            counts = record['action_counts']
            check(sum(counts) == horizon // 10 * 4 * n_agents, f'{label}: action coverage {u}')
            for metric in ('mean_acquired', 'mean_score_difference', 'mean_red_score_integral'):
                check(math.isfinite(record[metric]), f'{label}: nonfinite metric {metric} {u}')
            if m['scenario'].startswith('pickup_'):
                maximum = 6 if m['scenario'] == 'pickup_simple' else 24
                check(all(0 <= value <= maximum for value in record['acquired_per_world']), f'{label}: pickup conservation {u}')
                check(sum(counts[4:7]) == 0, f'{label}: pickup action mask {u}')
                check(all(v == 0 for v in record['score_blue_per_world'] + record['score_red_per_world']), f'{label}: pickup scoring {u}')
            if m['scenario'] == 'offense_vs_deterministic_defense':
                check(counts[5] == counts[6] == 0, f'{label}: offense oracle/ferry action {u}')
        check(changed, f'{label}: no learned parameter changes')
        training = [json.loads(line) for line in (directory / 'training.jsonl').read_text().splitlines() if line]
        check([row['update'] for row in training] == [1, 2, 3, 4], f'{label}: training coverage')
        check(all(math.isfinite(row[k]) for row in training for k in ('mean_loss', 'mean_reward')), f'{label}: nonfinite training')
        records.append({'case': label, 'source_sha256': m['provenance']['source_sha256'],
                        'parameters_changed': changed, 'training_acquisitions': sum(r['acquisitions'] for r in training),
                        'training_blue_score': sum(r['score_blue'] for r in training),
                        'training_red_score': sum(r['score_red'] for r in training),
                        'final_mean_acquired': curve[-1]['mean_acquired'],
                        'final_mean_score_difference': curve[-1]['mean_score_difference']})
    missing, extra = sorted(expected - found), sorted(found - expected)
    check(not missing, f'Missing {len(missing)} declared cases')
    check(not extra, f'Unexpected {len(extra)} cases')
    return {'passed': not errors, 'expected_cases': len(expected), 'found_cases': len(found),
            'source_sha256': expected_source, 'errors': errors, 'missing': missing, 'extra': extra,
            'checks': ['complete 105 main + 14 learning-rate cases', 'frozen source and checkpoint hashes',
                       'finite weights, losses and metrics', 'actual parameter updates',
                       'fixed train/evaluation seeds and budgets', 'real 20ms simulator and perception enabled',
                       'evaluation frame endpoints and decision counts', 'pickup conservation/action masks',
                       'experimental offense oracle/ferry actions disabled'],
            'cases': records}


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text())['source_sha256']
    report = audit(args.root, source)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'missing', 'extra')}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
