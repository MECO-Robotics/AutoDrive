# 2026 REBUILT Gamepiece Training

GPU-vectorized training for FRC robots acquiring FUEL, carrying it to an active HUB, and scoring. The offensive policy covers both INTAKE and DELIVER objective selection, including evasion/counter-defense; the separately selectable DEFENSE policy reads the same visible field state and chooses where to contest. AD* plus the robot controller handles movement. Policies are scored on acquisitions, FUEL scored or denied, and simulated match points.

## Train on the WX 9100 GPUs

The project uses two independent game policies. Install the included user services, then start both policies:

```sh
mkdir -p ~/.config/systemd/user
cp deploy/systemd/frc-gamepiece-*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now frc-gamepiece-offense.service frc-gamepiece-defense.service
```

Offense runs on `cuda:0`; defense runs on `cuda:1`. Each service simulates complete 160 s matches (20 s autonomous plus 140 s teleop) over 8,000 physics steps and runs PPO updates in 40 generations against sampled scripted, AD*, current learned, historical, and selected diverse/strong opponents. Each generation evaluates on matched seeded scenarios and retains strong, diverse policy checkpoints; parameter mutation is not used. It resumes from the latest saved policy after a process failure. Training state and checkpoints go under `checkpoints/rebuilt-gamepiece-{offense,defense}/`. Prior 89-input/3-action weights are preserved as historical opponents; the current 137-input/8-action policies start with a fresh model when those incompatible weights are encountered, then subsequent generations continue from PPO-trained weights.

Check both runs and follow their logs:

```sh
systemctl --user status frc-gamepiece-offense.service frc-gamepiece-defense.service
journalctl --user -fu frc-gamepiece-offense.service
journalctl --user -fu frc-gamepiece-defense.service
```

The dashboard at `http://100.121.248.52:8765/` reports generation progress, game-level acquisition/scoring metrics, and playback with FUEL positions. Per-generation evaluation records include FUEL acquisition/scoring, cycle time, defensive outcomes, and score differential against each baseline and population member.

For a direct launch, install the tensor extra and run either strategic game task:

```sh
python3 -m pip install -e '.[tensor]'
python3 -m frc_defense.tensor_training train --task counter_defense --algorithm generational --architecture strategic_adstar --generations 40 --opponent-pool-size 8 --strong-checkpoints 2 --envs 24 --horizon 8000 --device cuda:0 --output checkpoints/rebuilt-gamepiece-offense
python3 -m frc_defense.tensor_training train --task defense --algorithm generational --architecture strategic_adstar --generations 40 --opponent-pool-size 8 --strong-checkpoints 2 --envs 24 --horizon 8000 --device cuda:1 --output checkpoints/rebuilt-gamepiece-defense
```

## Game model

- Both policies observe the same visible game state: robot poses, FUEL locations and possession, HUB activity, field geometry, and match time. The defender receives no hidden attacker goal.
- Intake is on each robot's local `+X` front. Only FUEL in the forward capture band can be acquired.
- Policies select game objectives; AD* plans and the controller follows the route.
- The match lasts 160 s (20 s autonomous plus 140 s teleop). Both HUBs are active in autonomous; in teleop, their active state alternates every 30 s based on the autonomous FUEL result. This is the requested training schedule, simplified from the official shift timing. A dumper scores only while facing its HUB within a 45-degree cone.
- Scoring, possession, pickup, and legal field contact update the simulation and the training reward. Game evaluations report acquisitions, scores, cycle time, defensive delay, denied/abandoned objectives, contacts, and total simulated score.

The model keeps gamepiece motion lightweight. HUB scoring is a planar proximity surrogate for FUEL passing through the regulation opening; it does not simulate launch trajectories, height, or the sensor array. Drivetrain parameters are illustrative unless populated from robot hardware and checked against measured traces. See the [official 2026 REBUILT manual](https://firstfrc.blob.core.windows.net/frc2026/Manual/HTML/2026GameManual.htm) and [FIRST field drawings](https://firstfrc.blob.core.windows.net/frc2026/FieldAssets/2026-field-dimension-dwgs.pdf).

## Host setup

The training units assume a Python 3.12 environment at `.venv/` with the pinned ROCm build in `requirements-rocm-gfx900.txt`. Install on this host with:

```sh
uv venv --python 3.12 .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install --no-cache-dir -r requirements-rocm-gfx900.txt
.venv/bin/python -m pip install --no-cache-dir -e .
```

Training checkpoints, evaluation data, logs, and local environment files are generated locally and excluded from Git.
