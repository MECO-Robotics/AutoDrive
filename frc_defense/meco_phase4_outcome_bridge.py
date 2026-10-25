"""Fail-closed diagnostic reducers for Phase 4 policy/behavior outcomes."""
from __future__ import annotations


def reduce_policy_family_outcomes(rows: list[dict], *,
        expected_policy_ids: set[str], categories: list[str], seeds: list[int],
        family_by_policy: dict[str, str]) -> dict:
    """Group exact paired whole-team outcomes by canonical assigned family.

    The returned associations are not per-policy causal contributions and do
    not define interception. Each input row must already be bound to the same
    evaluation source, checkpoint bundle, device, case, and paired reports.
    """
    if (not isinstance(rows, list) or not expected_policy_ids or not categories or
            not seeds or set(family_by_policy) != expected_policy_ids or
            any(not isinstance(family, str) or not family for family in
                family_by_policy.values())):
        raise ValueError("reactive family reducer requires the exact policy/category/seed domain")
    expected = {(policy, category, seed) for policy in expected_policy_ids
                for category in categories for seed in seeds}
    by_key = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("reactive family outcome row must be an object")
        key = (row.get("policy_id"), row.get("category"), row.get("seed"))
        if key not in expected or key in by_key:
            raise ValueError("reactive family outcomes contain an unexpected or duplicate key")
        if (row.get("policy_family") != family_by_policy[key[0]] or
                not isinstance(row.get("control_episode_outcome"), dict) or
                not isinstance(row.get("treatment_episode_outcome"), dict) or
                row.get("outcome_semantics") != (
                    "whole-team paired episode outcome while the policy was assigned; "
                    "association by policy family, not per-policy causal contribution")):
            raise ValueError("reactive family outcome row has mismatched family or semantics")
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError("reactive family outcomes lack complete policy/category/seed coverage")
    grouped = {}
    for key in sorted(expected):
        grouped.setdefault(family_by_policy[key[0]], []).append(by_key[key])
    return {"policy_family_coverage": {family: {
                "policy_ids": sorted({row["policy_id"] for row in family_rows}),
                "paired_episode_rows": len(family_rows)}
                for family, family_rows in sorted(grouped.items())},
            "paired_outcome_by_family": grouped,
            "paired_outcome_semantics": (
                "whole-team paired episode outcomes grouped by assigned policy family; "
                "association only, not an interception or individual causal contribution")}


def reduce_teammate_behavior_outcomes(rows: list[dict], *,
        expected_keys: set[tuple[str, str, int, str]],
        expected_behaviors: set[str]) -> dict:
    """Require exact paired whole-team outcomes for every assigned behavior."""
    if not isinstance(rows, list) or not expected_keys or not expected_behaviors:
        raise ValueError("teammate behavior reducer requires a nonempty exact pair domain")
    keys = set()
    grouped = {behavior: [] for behavior in expected_behaviors}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("teammate behavior outcome row must be an object")
        key = (row.get("policy_id"), row.get("category"), row.get("seed"),
               row.get("behavior"))
        behavior = row.get("behavior")
        if key not in expected_keys or behavior not in expected_behaviors or key in keys:
            raise ValueError("teammate behavior outcomes contain duplicate or unexpected rows")
        if (not isinstance(row.get("control_whole_team_outcome"), dict) or
                not isinstance(row.get("treatment_whole_team_outcome"), dict) or
                row.get("outcome_semantics") != (
                    "paired whole-team episode outcomes while this behavior was present; "
                    "concurrent behavior slots prevent individual score attribution")):
            raise ValueError("teammate behavior outcome row has invalid metrics or semantics")
        keys.add(key)
        grouped[behavior].append(row)
    if keys != expected_keys or any(not grouped[behavior] for behavior in expected_behaviors):
        raise ValueError("teammate behavior outcomes lack exact paired policy/category/seed coverage")
    return {"outcome_by_teammate_behavior": grouped,
        "teammate_collision_rate": None, "score_contribution": None,
        "unavailable_signal_sources": {
            "teammate_collision_rate":
                "no slot-pair-specific simulator collision ledger is exposed",
            "score_contribution":
                "simulator reports team-level FUEL metrics without robot-slot attribution"}}
