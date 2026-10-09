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

Free FUEL uses individual 3D spheres with gravity, compliant contacts, sliding friction, rolling resistance, and equal/opposite robot contact impulses. Robot and fuel physics advance together with a baseline of ten 2 ms substeps per 20 ms controller tick, refined for fast motion; observations retain their existing planar position/velocity interface. Fuel follows the same triangular bump terrain profile as the drivetrain. The 3v3 collector predicts short intercepts using observed fuel velocity, and its pickup bins follow moving fuel. Ferry passes respawn moving balls at resolved ground landings; scored-fuel returns use illustrative ballistic settings. HUB score detection remains a game-rule surrogate, rather than a sensor-array simulation.

Fuel contact coefficients and launch settings require calibration against measured roll, bounce, and pile-pushing traces. The rolling resistance and impact damping defaults are tuned to settle free FUEL and dissipate most collision energy on the modeled carpet. Robot chassis dynamics remain planar: wheel climbing over fuel, chassis lift, suspension, and detailed foam deformation are not resolved. Drivetrain parameters are also illustrative unless populated from robot hardware and checked against measured traces. See the [official 2026 REBUILT manual](https://firstfrc.blob.core.windows.net/frc2026/Manual/HTML/2026GameManual.htm) and [FIRST field drawings](https://firstfrc.blob.core.windows.net/frc2026/FieldAssets/2026-field-dimension-dwgs.pdf).

### Fuel physics configuration

Ferry releases and scored-fuel returns use 10% of the nominal horizontal respawn velocity, giving 1% of the nominal horizontal kinetic energy. Ferry fuel appears on the terrain at its landing point with this residual speed. Scored-fuel returns retain their flight timing. Floor friction and rolling resistance continue slowing the fuel after contact.

Scored-fuel returns sample fresh directions in a 45° total cone (±22.5°) centered from their HUB toward midfield. This changes direction while preserving the reduced release speed.

Ferry landings sample a 10° total cone (±5°) toward a safe point in the friendly alliance zone. They spawn on the terrain with low residual velocity away from the robot. Out-of-bounds landings appear at the first field edge crossed, inset by the fuel radius. A path crossing the friendly HUB's midfield-facing edge between the two bumps returns fuel to midfield with the scored-ball ballistic launch settings.

Both tensor environments accept `fuel_physics=True` (default), `False` for the previous static-fuel behavior, or a dictionary of `FuelPhysicsConfig` options. For example:

```python
env = TensorThreeVsThreeEnv(
    num_envs=24, device="cuda:0",
    fuel_physics={"substeps": 10, "broadphase": "grid", "cache": True,
                  "contact_mode": "fused", "solver": "jacobi", "sleep": True},
)
```

The two-robot environment also accepts `gamepieces.physics` in `FRC_DRIVETRAIN_CONFIG`. The all-pairs path is a collision reference; cached grid neighbors retain conservative motion margins and never silently drop contacts on capacity overflow. Robot contacts default to scanning the small robot set, which measured faster than `robot_broadphase="grid"` at six robots and 504 fuel pieces on both GPUs. `sleep_islands=True` enables sleeping and waking connected supported fuel groups. Adaptive substeps use one host speed reduction per controller tick; graph comparisons use an explicitly fixed timestep and a checked motion envelope. Backend and variant comparisons, quality diagnostics, and measured device timings are recorded in `docs/FUEL_PHYSICS_BENCHMARKS.md` by `scripts/benchmark_fuel_physics.py`.

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
