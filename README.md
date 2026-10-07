# 2026 REBUILT Gamepiece Training

GPU-vectorized training for FRC robots acquiring FUEL, carrying it to an active HUB, and scoring. Offense uses the deterministic game strategy; PPO trains defense to contest using the same visible field state. AD* plus the robot controller handles movement. Defense is scored on FUEL denied and simulated match points.

## Train on the WX 9100 GPUs

Install and start the defense training service:

```sh
mkdir -p ~/.config/systemd/user
cp deploy/systemd/frc-gamepiece-*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now frc-gamepiece-defense.service
```

Defense trains on complete 160 s matches (20 s autonomous plus 140 s teleop) over 8,000 physics steps. PPO uses deterministic offense opponents and evaluates against fixed scripted strategies on matched seeded scenarios. The service resumes from the latest compatible defense checkpoint after a process failure. Training state and checkpoints go under `checkpoints/rebuilt-gamepiece-defense/`.

Check the training service and follow its log:

```sh
systemctl --user status frc-gamepiece-defense.service
journalctl --user -fu frc-gamepiece-defense.service
```

The dashboard at `http://100.121.248.52:8765/` reports generation progress, game-level acquisition/scoring metrics, and playback with FUEL positions. Per-generation evaluation records include FUEL acquisition/scoring, cycle time, defensive outcomes, and score differential against each deterministic baseline.

For a direct launch, install the tensor extra and start defense training:

```sh
python3 -m pip install -e '.[tensor]'
python3 -m frc_defense.tensor_training train --task defense --algorithm generational --architecture strategic_adstar --generations 40 --opponent-pool-size 8 --strong-checkpoints 2 --envs 24 --horizon 8000 --device cuda:1 --output checkpoints/rebuilt-gamepiece-defense
```

## Game model

- The defender observes visible game state: robot poses, FUEL locations and possession, HUB activity, field geometry, and match time. It receives no hidden attacker goal.
- Intake is on each robot's local `+X` front. Only FUEL in the forward capture band can be acquired.
- The deterministic offense and learned defense select game objectives; AD* plans and the controller follows the route.
- The match lasts 160 s (20 s autonomous plus 140 s teleop). Both HUBs are active in autonomous; in teleop, their active state alternates every 30 s based on the autonomous FUEL result. This is the requested training schedule, simplified from the official shift timing. A dumper scores only while facing its HUB within a 45-degree cone.
- Scoring, possession, pickup, and legal field contact update the simulation and the training reward. Game evaluations report acquisitions, scores, cycle time, defensive delay, denied/abandoned objectives, contacts, and total simulated score.

The model keeps gamepiece motion lightweight. HUB scoring is a planar proximity surrogate for FUEL passing through the regulation opening; it does not simulate launch trajectories, height, or the sensor array. Drivetrain parameters are illustrative unless populated from robot hardware and checked against measured traces. See the [official 2026 REBUILT manual](https://firstfrc.blob.core.windows.net/frc2026/Manual/HTML/2026GameManual.htm) and [FIRST field drawings](https://firstfrc.blob.core.windows.net/frc2026/FieldAssets/2026-field-dimension-dwgs.pdf).

### Swerve physics assumptions

The tensor drivetrain models four independently steered wheels. Module velocity targets are generated from chassis translation and rotation, optimized to keep each steering move within 90 degrees, and cosine-compensated while the wheel is turning. Drive motor voltage, back-EMF, current limits, battery sag, wheel torque, and a per-wheel friction circle constrain the resulting chassis forces. Chassis speed, acceleration, yaw rate, and yaw acceleration are additional configurable caps. Bump geometry adds wheel load transfer and grade forces; suspension, wheel lift, thermal behavior, and detailed controller timing are outside the model.

Built-in motor constants are a Kraken X60 nominal motor curve, with an illustrative 6.75:1 drive ratio and 4-inch wheel radius. At 6,000 motor RPM, those values give about 4.73 m/s ideal wheel speed at 12 V; the chassis cap is 4.8 m/s and the motor voltage/back-EMF model remains the limiting factor. Rotating and translating together is limited by desaturating the four actual module velocity vectors, rather than subtracting the worst-case rotational speed from the chassis translation cap. The 55 kg robot mass is a nominal configurable assumption. These numbers are not a claim that every FRC robot uses this hardware. Supply a robot-specific JSON file through `FRC_DRIVETRAIN_CONFIG` to override `mass`, chassis geometry and limits, friction, and `swerve` motor, ratio, current, wheel, and module-offset parameters. Set `randomize: true` only when variation around those nominal values is desired.

The control details follow [WPILib's swerve kinematics guidance](https://docs.wpilib.org/en/latest/docs/software/kinematics-and-odometry/swerve-drive-kinematics.html), including optimized module states and cosine compensation. The nominal 6,000 RPM and 7.09 Nm motor values match published [Kraken X60 specifications](https://store.ctr-electronics.com/products/kraken-x60); CTRE recommends using battery voltage and a motor-specific simulation model for higher fidelity ([simulation guidance](https://pro.docs.ctr-electronics.com/en/stable/docs/api-reference/simulation/simulation-intro.html)). A mechanism ratio and wheel size must be matched to the selected module, such as the [SDS MK4i module options](https://www.andymark.com/products/mk4i-swerve-modules). Field friction and steering response are especially robot-dependent, so calibration against measured robot acceleration, coast-down, and steering traces remains necessary for hardware-faithful results.

## Host setup

The training units assume a Python 3.12 environment at `.venv/` with the pinned ROCm build in `requirements-rocm-gfx900.txt`. Install on this host with:

```sh
uv venv --python 3.12 .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install --no-cache-dir -r requirements-rocm-gfx900.txt
.venv/bin/python -m pip install --no-cache-dir -e .
```

Training checkpoints, evaluation data, logs, and local environment files are generated locally and excluded from Git.
