# MECO opponent prediction mode

MECO Plan A, Plan C, D1, and D2 now use the shared `current|cv` prediction contract in `frc_defense/meco_tracks.py`.

- Learned A/C training defaults to `cv`, preserving its prior route-arrival forecast. `current` uses the finite sensed opponent pose. `cv` uses the observed velocity capped at 3 m/s when the track age is at most 0.5 seconds; otherwise it falls back to the finite sensed pose. A missing or nonfinite pose adds no risk penalty.
- Candidate-based policies forecast to each candidate's route ETA. Plan A forecasts separately to each graph node's route ETA. Those horizons, observed age, forecast use, and fallback are retained in candidate route metadata or the selected Plan A route record.
- D1 defaults to `current`; D2 defaults to `cv`. The MECO evaluator accepts `--prediction-mode current|cv` to run both controls with the same mode. Without that override, the historical D1/D2 control mapping is preserved.
- The mode, forecast contract, and training configuration digest are stored with each checkpoint/run; evaluation records include control and learned modes. These additions do not alter the frozen qualification protocol artifact.

Example: `python3 -m frc_defense.meco_track_training evaluate ... --prediction-mode current`.

Evaluation reports also include synchronized component wall timings with p50/p95/p99 and sample counts. The decision pipeline combines candidate/observation preparation, model work, and scoring; it is not an inference-only measurement. CUDA/ROCm runs synchronize around each measured component, and no unreported simulator warmup is performed.

For an eval-only same-seed inference fault pair, pass `--inference-disable-tick N` with `0 < N < scenario horizon`. Each selected learned track gets a normal record and a fault record. The fault takes effect at the first macro decision boundary at or after physics tick `N`; if the episode ends first, the record says the trigger was not reached. At activation, Plan A's pending goal, C1/C2 commitment, and hidden state are cleared. The remaining decisions use the deterministic D1 scorer with that checkpoint's prediction mode. Records retain pre/post actions, a shared pair ID, fallback count, post-fault progress/outcome deltas, world-wide robot-contact ticks, and controlled slot-3 field/wall-contact ticks. Trigger activation does not imply recovery or a successful episode.

The former standalone `curriculum_training evaluate --policy d2` diagnostic retains its fixed one-second D2 behavior and separate identity.
