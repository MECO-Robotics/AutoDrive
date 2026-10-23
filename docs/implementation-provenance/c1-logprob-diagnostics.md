# C1 PPO rollout log-probability diagnostic tranche

Evidence type: transcript summary materialized on 2026-10-23; this is not an original persisted agent artifact.

- Requirement scope: diagnose the C1 failure at live source HEAD `b23b9197a`, update 34, where `Plan C rollout log-probability does not replay under frozen candidate table` occurred. Training reproduction remains pending.
- Planner: `/root/next_gap_planner` traced sampled-row candidate table/log-probability recomputation and recommended recording diagnostic details before changing the consistency check. Original artifact path: not persisted; this summary captures the reported finding.
- Implementer: diagnostic recording `ac6681e3b`; correction `d7dc22266`. Path: `frc_defense/meco_track_training.py`.
- Reviewer: initial review blocked over scalar hash/write mask; exact corrected commit `d7dc22266` approved.
- Validation limit: training reproduction pending in the active session; no claim of a repaired five-track run.
