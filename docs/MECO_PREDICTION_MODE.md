# MECO opponent prediction mode

MECO Plan A, Plan C, D1, and D2 use the shared `current|cv|multi_model_v1` prediction contract in `frc_defense/meco_tracks.py`. The new mode is opt-in; the existing defaults and checkpoint identities are unchanged.

- Learned A/C training defaults to `cv`, preserving its prior route-arrival forecast. `current` uses the finite sensed opponent pose. `cv` uses the observed velocity capped at 3 m/s when the track age is at most 0.5 seconds; otherwise it falls back to the finite sensed pose. A missing or nonfinite pose adds no risk penalty.
- Candidate-based policies forecast to each candidate's route ETA. Plan A forecasts separately to each graph node's route ETA. Those horizons, observed age, forecast use, and fallback are retained in candidate route metadata or the selected Plan A route record.
- D1 defaults to `current`; D2 defaults to `cv`. The MECO evaluator accepts `--control-prediction-mode current|cv|multi_model_v1` to run both controls with the same mode. Without that override, the historical D1/D2 control mapping is preserved.
- `multi_model_v1` is a versioned deterministic mixture of four kinematic hypotheses: constant velocity (0.40), left turn (0.15), right turn (0.15), and exponentially damped velocity (0.30). Velocity is capped at 3 m/s, track age at 0.5 s, and forecast horizon at 2.5 s. Turn rate is clamped to 0.6–1.5 rad/s using the observed angular rate magnitude when available, with 0.6 rad/s as the fallback; the two turn directions remain explicit hypotheses. Risk is the weighted expected proximity penalty across the endpoints. Stale or invalid motion falls back to a finite observed pose; a missing/nonfinite pose contributes no prediction risk. It is a deterministic kinematic diagnostic, not an intent, route-feasibility, collision, or calibrated uncertainty model.
- The evaluator accepts `--prediction-mode multi_model_v1` to apply the same forecast to all selected learned tracks and `--control-prediction-mode multi_model_v1` to match D1 and D2 in a control evaluation. Matrix conditions bind both fields and the full predictor contract. Each route/node uses its own arrival-time horizon, capped under this mode.
- The mode, forecast contract, and training configuration digest are stored with each checkpoint/run; evaluation records include control and learned modes. These additions do not alter the frozen qualification protocol artifact.

Examples:

```bash
python3 -m frc_defense.meco_track_training evaluate ... --prediction-mode current
python3 -m frc_defense.meco_track_training evaluate ... --prediction-mode multi_model_v1 --control-prediction-mode multi_model_v1
```

Evaluation reports also include synchronized component wall timings with p50/p95/p99 and sample counts. The decision pipeline combines candidate/observation preparation, model work, and scoring; it is not an inference-only measurement. CUDA/ROCm runs synchronize around each measured component, and no unreported simulator warmup is performed.

For an eval-only same-seed inference fault pair, pass `--inference-disable-tick N` with `0 < N < scenario horizon`. Each selected learned track gets a normal record and a fault record. The fault takes effect at the first macro decision boundary at or after physics tick `N`; if the episode ends first, the record says the trigger was not reached. At activation, Plan A's pending goal, C1/C2 commitment, and hidden state are cleared. The remaining decisions use the deterministic D1 scorer with that checkpoint's prediction mode. Records retain pre/post actions, a shared pair ID, fallback count, post-fault progress/outcome deltas, world-wide robot-contact ticks, and controlled slot-3 field/wall-contact ticks. Trigger activation does not imply recovery or a successful episode.

The former standalone `curriculum_training evaluate --policy d2` diagnostic retains its fixed one-second D2 behavior and separate identity.

The new predictor has not been calibrated against held-out opponent trajectories or qualified through a simulator campaign. Existing decision timing measures the combined policy pipeline; isolated prediction-component timing remains a separate evaluation-plan gap.
