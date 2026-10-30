from pathlib import Path
import json
import os
from frc_defense.meco_track_training import evaluate_meco

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'profiling/combined-sim-accounting-e2e'
SEED = 2970053994459693056
os.environ['AUTODRIVE_EXPERIMENTAL_CONTACT_ACCOUNTING_FUSION'] = '1'
os.environ['AUTODRIVE_EXPERIMENTAL_FUSED_SIM_TICK_ACCOUNTING'] = '1'
OUT.mkdir(parents=True, exist_ok=True)
report_path = OUT / 'evaluation.json'
replay_dir = OUT / '.scenario-progress'
report = evaluate_meco(
    {}, scenario_id='full_match_offense',
    categories=('active/open/level_5_mixed',), episodes=1,
    master_seed=SEED, episode_seeds=(SEED,), device='cuda:0',
    include_controls=True, control_policies=('D0',),
    evaluation_policies=(), split='diagnostic',
    output=str(report_path), replay_output_dir=replay_dir,
    replay_selection={'D0': [SEED]},
)
summary = {
    'evaluation_id': report.get('evaluation_id'),
    'scenario': 'full_match_offense', 'category': 'active/open/level_5_mixed',
    'policy': 'D0', 'seed': SEED,
    'evaluation': str(report_path), 'replay_dir': str(replay_dir),
    'records': report.get('records'),
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
