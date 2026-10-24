# D1 versus D2 multi-model matrix: Planner summary

Scope: make the plan's D1 deterministic-current-prediction versus D2 multi-model-kinematics comparison expressible as one paired matrix condition. The earlier scalar control-mode override applied the same forecast to both controls, so it could not isolate the predictor contribution. This tranche adds a per-control mode binding and keeps the existing D1=current/D2=CV comparison. It does not claim a completed episode campaign or measured predictive benefit.
