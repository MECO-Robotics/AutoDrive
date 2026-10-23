#!/usr/bin/env python3
"""Fair, source-frozen curriculum workers; dry-run unless --run is supplied.

Each model/seed keeps its optimizer and curriculum state across bounded update
chunks. A failed lane remains visible while other lanes continue. Restarting
the supervisor resumes committed checkpoints; failed lanes require --retry-failed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

REPO = Path(__file__).resolve().parents[1]
ARCHITECTURES = ('baseline_mlp', 'residual_mlp', 'deep_sets', 'entity_attention',
                 'full_field_cnn', 'full_field_resnet', 'field_fno')


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def source_hash():
    files = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((REPO/'frc_defense').rglob('*'))
             if p.suffix in ('.py', '.cpp', '.cu', '.h', '.cuh', '.json', '.hip', '.hpp', '.cc')}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def build_lanes(output, seeds, devices, algorithms=('ppo',), architectures=ARCHITECTURES):
    if (not algorithms or len(set(algorithms)) != len(algorithms) or
            any(algorithm not in ('ppo', 'sac') for algorithm in algorithms)):
        raise ValueError('algorithms must be a nonempty selection of ppo and sac')
    if not architectures or len(set(architectures)) != len(architectures) or any(
            architecture not in ARCHITECTURES for architecture in architectures):
        raise ValueError('architectures must be a nonempty selection of supported models')
    return [{'algorithm': algorithm, 'architecture': architecture, 'seed': seed,
             'device': devices[(i+j) % len(devices)],
             'directory': str((output/'variants'/algorithm/architecture/f'seed-{seed}').resolve()),
             'status': 'pending', 'completed_updates': 0, 'qualified': []}
            for j, seed in enumerate(seeds) for algorithm in algorithms
            for i, architecture in enumerate(architectures)]


def training_command(config, lane, target):
    command = [config['python'], '-m', 'frc_defense.curriculum_training', 'train',
               '--algorithm', lane.get('algorithm', config.get('algorithm', 'ppo')),
               '--architecture', lane['architecture'], '--updates', str(target),
               '--learning-rate', str(config['learning_rate']),
               '--learning-rate-start', str(config['learning_rate_start']),
               '--learning-rate-decay-updates', str(config['learning_rate_decay_updates']),
               '--seed', str(lane['seed']), '--device', lane['device'],
               '--output', lane['directory'], '--envs', str(config['envs']),
               '--rollout-steps', str(config['rollout_steps']), '--epochs', str(config['epochs']),
               '--eval-every', str(config['eval_every']),
               '--evaluation-episodes', str(config['evaluation_episodes']),
               '--action-ticks', str(config['action_ticks']), '--disable-miopen']
    command += ['--minibatch-size', str(config['minibatch_size'])]
    if lane.get('algorithm', config.get('algorithm', 'ppo')) == 'sac':
        for option, key in (('replay-capacity', 'sac_replay_capacity'),
                            ('batch-size', 'sac_batch_size'),
                            ('warmup-steps', 'sac_warmup_steps'),
                            ('updates-per-transition', 'sac_updates_per_transition'),
                            ('target-tau', 'sac_target_tau'),
                            ('initial-alpha', 'sac_initial_alpha'),
                            ('target-entropy-fraction', 'sac_target_entropy_fraction')):
            command += [f'--sac-{option}', str(config[key])]
    checkpoint = Path(lane['directory'])/'latest.pt'
    if checkpoint.exists():
        command += ['--resume', str(checkpoint)]
    return command


def committed_progress(lane):
    checkpoint = Path(lane['directory'])/'latest.pt'
    if not checkpoint.exists():
        return 0, [], 'pickup_simple'
    import torch
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['config'].get('architecture', 'baseline_mlp') != lane['architecture']:
        raise ValueError('checkpoint architecture differs from lane')
    if payload['config'].get('algorithm', 'ppo') != lane.get('algorithm', 'ppo'):
        raise ValueError('checkpoint algorithm differs from lane')
    if payload['config']['seed'] != lane['seed'] or payload['config']['device'] != lane['device']:
        raise ValueError('checkpoint seed/device differs from lane')
    return payload['update'], payload['curriculum_state']['qualified'], payload['current']


def run_queue(output, configuration, *, expected_source, retry_failed=False):
    output.mkdir(parents=True, exist_ok=True)
    lockfile = (output/'queue.lock').open('a')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = output/'orchestration.json'
    if source_hash() != expected_source:
        raise ValueError('runtime source does not match --expected-source')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['configuration'] != configuration or manifest['source_identity_sha256'] != expected_source:
            raise ValueError('queue configuration/source differs; use the existing exact configuration or a new output')
        for lane in manifest['lanes']:
            pid = lane.get('pid')
            if lane['status'] == 'running' and pid and Path(f'/proc/{pid}').exists():
                raise ValueError(f'previous child {pid} is still alive; refusing duplicate training')
            if lane['status'] != 'failed' or retry_failed:
                lane['status'] = 'pending'
    else:
        manifest = {'configuration': configuration, 'source_identity_sha256': expected_source,
                    'lanes': build_lanes(output, configuration['seeds'], configuration['devices'],
                                         configuration.get('algorithms', ('ppo',)),
                                         configuration.get('architectures', ARCHITECTURES)),
                    'attempts': []}
    stop = threading.Event()
    mutex = threading.Lock()
    manifest.update(status='running', supervisor_pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat())

    def save():
        manifest['updated_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['updated_at'] = manifest['updated_utc']
        atomic_json(manifest_path, manifest)

    def report():
        script = REPO/'scripts/report_curriculum_progress.py'
        if script.exists():
            with (output/'reporter.log').open('a') as handle:
                result = subprocess.run([configuration['python'], str(script), '--root', str(output),
                                         '--output', str(output/'report/progress.json')],
                                        cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, timeout=45)
            with mutex:
                manifest['reporter_returncode'] = result.returncode
                save()

    def terminate(signum, frame):
        stop.set()

    handlers = {s: signal.signal(s, terminate) for s in (signal.SIGTERM, signal.SIGINT)}
    save()

    def worker(device):
        lanes = [lane for lane in manifest['lanes'] if lane['device'] == device]
        while not stop.is_set():
            work = False
            for lane in lanes:
                if stop.is_set():
                    break
                if lane['status'] in ('failed', 'complete', 'budget_complete'):
                    continue
                work = True
                process = None
                attempt = None
                try:
                    if source_hash() != expected_source:
                        stop.set()
                        raise ValueError('runtime source changed; queue stopped before further training')
                    completed, qualified, current = committed_progress(lane)
                    maximum = configuration['max_updates']
                    target = min(completed+configuration['chunk_updates'], maximum)
                    with mutex:
                        lane.update(completed_updates=completed, qualified=qualified, current=current)
                        if completed >= maximum:
                            lane['status'] = 'budget_complete'
                            save()
                            continue
                        number = len(manifest['attempts'])+1
                        log = output/'logs'/f'{number:06d}-{lane["algorithm"]}-{lane["architecture"]}-seed-{lane["seed"]}.log'
                        log.parent.mkdir(exist_ok=True)
                        command = training_command(configuration, lane, target)
                        attempt = {'algorithm': lane['algorithm'], 'architecture': lane['architecture'], 'seed': lane['seed'], 'device': device,
                                   'from_update': completed, 'target_updates': target, 'command': command,
                                   'log': str(log), 'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat()}
                        manifest['attempts'].append(attempt)
                        lane.update(status='running', log=str(log), last_attempt=number)
                        save()
                    environment = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
                                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                    with log.open('x') as handle:
                        process = subprocess.Popen(command, cwd=REPO, env=environment,
                                                   stdout=handle, stderr=subprocess.STDOUT)
                        with mutex:
                            attempt['pid'] = lane['pid'] = process.pid
                            save()
                        reported_at = time.monotonic()
                        while process.poll() is None:
                            if stop.wait(1):
                                process.terminate()
                                try:
                                    process.wait(timeout=15)
                                except subprocess.TimeoutExpired:
                                    process.kill(); process.wait()
                                break
                            if time.monotonic()-reported_at >= 30:
                                with mutex:
                                    lane['log_bytes'] = log.stat().st_size
                                    lane['process_alive_utc'] = datetime.now(timezone.utc).isoformat()
                                    stats_path = Path(lane['directory'])/'stats.json'
                                    if stats_path.exists():
                                        stats = json.loads(stats_path.read_text())
                                        lane['observed_stats_updates'] = stats.get('updates')
                                        lane['qualified'] = stats.get('curriculum_state', {}).get('qualified', [])
                                    save()
                                try:
                                    with report_lock:
                                        report()
                                except Exception as error:
                                    with mutex:
                                        manifest['reporter_error'] = str(error)
                                        save()
                                reported_at = time.monotonic()
                    with mutex:
                        attempt.update(returncode=process.returncode, ended_utc=datetime.now(timezone.utc).isoformat())
                        lane.pop('pid', None)
                    if stop.is_set():
                        with mutex:
                            lane['status'] = attempt['status'] = 'interrupted'
                            save()
                        break
                    if process.returncode:
                        raise RuntimeError(f'trainer exited {process.returncode}; see {log}')
                    if source_hash() != expected_source:
                        stop.set()
                        raise ValueError('runtime source changed during training')
                    finished, qualified, current = committed_progress(lane)
                    if finished != target:
                        raise RuntimeError(f'expected committed update {target}, got {finished}')
                    with mutex:
                        lane.update(completed_updates=finished, qualified=qualified, current=current,
                                    status='budget_complete' if finished >= maximum else 'pending')
                        attempt['status'] = 'complete'
                        save()
                except Exception as error:
                    with mutex:
                        lane.update(status='failed', error=str(error))
                        if attempt is not None and attempt.get('status') == 'running':
                            attempt.update(status='failed', error=str(error))
                        save()
                try:
                    # One reporter at a time: atomic files are shared between workers.
                    with report_lock:
                        report()
                except Exception as error:
                    with mutex:
                        manifest['reporter_error'] = str(error)
                        save()
            if not work:
                return

    report_lock = threading.Lock()
    try:
        with ThreadPoolExecutor(max_workers=len(configuration['devices'])) as pool:
            futures = [pool.submit(worker, device) for device in configuration['devices']]
            while any(not f.done() for f in futures):
                time.sleep(2)
                with mutex:
                    save()
            for future in futures:
                future.result()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        with mutex:
            manifest['status'] = ('interrupted' if stop.is_set() else
                                  'completed_with_failures' if any(l['status']=='failed' for l in manifest['lanes']) else 'budget_complete')
            save()
        lockfile.close()
    return 130 if stop.is_set() else 1 if manifest['status']=='completed_with_failures' else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=REPO/'.venv/bin/python')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 43])
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'])
    parser.add_argument('--algorithms', nargs='+', choices=('ppo', 'sac'), default=['ppo'])
    parser.add_argument('--architectures', nargs='+', choices=ARCHITECTURES, default=list(ARCHITECTURES))
    for name, default in [('chunk-updates', 5), ('max-updates', 2000), ('envs', 8),
                          ('rollout-steps', 16), ('epochs', 4), ('eval-every', 25),
                          ('evaluation-episodes', 100), ('action-ticks', 12), ('minibatch-size', 64)]:
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--learning-rate-start', type=float, default=3e-3)
    parser.add_argument('--learning-rate-decay-updates', type=int, default=75)
    for name, kind, default in (
            ('replay-capacity', int, 4096), ('batch-size', int, 256),
            ('warmup-steps', int, 1024), ('updates-per-transition', float, .25),
            ('target-tau', float, .005), ('initial-alpha', float, .2),
            ('target-entropy-fraction', float, .98)):
        parser.add_argument('--sac-'+name, type=kind, default=default)
    parser.add_argument('--expected-source')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if any(getattr(args, k)<1 for k in ('chunk_updates', 'max_updates', 'envs', 'rollout_steps',
                                       'epochs', 'eval_every', 'evaluation_episodes', 'action_ticks', 'minibatch_size')):
        parser.error('all budget/rate settings must be positive')
    if len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0 or len(set(args.devices))!=len(args.devices):
        parser.error('seeds/devices must be unique and seeds nonnegative')
    if len(set(args.algorithms)) != len(args.algorithms):
        parser.error('algorithms must be unique')
    if len(set(args.architectures)) != len(args.architectures):
        parser.error('architectures must be unique')
    if (args.learning_rate <= 0 or args.learning_rate_start < args.learning_rate or
            args.learning_rate_decay_updates < 1):
        parser.error('learning-rate start must be at least the positive floor and decay length positive')
    configuration = {k:v for k,v in vars(args).items() if k not in ('output','run','expected_source','retry_failed')}
    configuration['python'] = str(args.python.absolute())
    if not args.run:
        print(json.dumps({'configuration': configuration, 'source_identity_sha256': source_hash(),
                          'lanes': build_lanes(args.output, args.seeds, args.devices,
                                               args.algorithms,args.architectures)}, indent=2))
        return 0
    if not args.expected_source:
        parser.error('--run requires --expected-source from the reviewed dry-run')
    return run_queue(args.output.resolve(), configuration, expected_source=args.expected_source,
                     retry_failed=args.retry_failed)


if __name__ == '__main__':
    raise SystemExit(main())
