"""Immutable, versioned interchange types for experimental learned controllers.

This module deliberately does not alter the curriculum's 137-feature, eight
action policy contract. Values are tuples so callers cannot mutate a snapshot
after publishing it to an asynchronous scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


def _finite_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} values must be finite")
    return result


@dataclass(frozen=True, slots=True)
class ObservationSnapshot:
    """Versioned pose-graph observation; raster is channel-major and flattened."""

    snapshot_id: str
    episode_id: str
    sequence: int
    timestamp: float
    raster_shape: tuple[int, int, int]
    raster: tuple[float, ...]
    globals: tuple[float, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.snapshot_id or not self.episode_id:
            raise ValueError("snapshot_id and episode_id must be nonempty")
        if self.schema_version != 1:
            raise ValueError("unsupported observation schema_version")
        if self.sequence < 0 or not math.isfinite(self.timestamp):
            raise ValueError("sequence and timestamp must be nonnegative/finite")
        if len(self.raster_shape) != 3 or any(size <= 0 for size in self.raster_shape):
            raise ValueError("raster_shape must contain three positive dimensions")
        expected = math.prod(self.raster_shape)
        raster = _finite_tuple(self.raster, "raster")
        globals_ = _finite_tuple(self.globals, "globals")
        if len(raster) != expected:
            raise ValueError(f"raster has {len(raster)} values, expected {expected}")
        if len(globals_) != 7:
            raise ValueError("globals must contain exactly seven values")
        object.__setattr__(self, "raster", raster)
        object.__setattr__(self, "globals", globals_)
        object.__setattr__(self, "raster_shape", tuple(self.raster_shape))


@dataclass(frozen=True, slots=True)
class CandidateTable:
    """Stable ordered action-trace rows with named, nullable features."""

    snapshot_id: str
    candidate_ids: tuple[str, ...]
    features: tuple[tuple[float | None, ...], ...]
    valid_mask: tuple[bool, ...]
    feature_names: tuple[str, ...]
    route_costs: tuple[float | None, ...] | None = None
    schema_version: int = 2
    context_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 2 or not self.snapshot_id:
            raise ValueError("unsupported schema or empty snapshot_id")
        if self.context_digest and (len(self.context_digest) != 64 or any(
                char not in "0123456789abcdef" for char in self.context_digest)):
            raise ValueError("context_digest must be an empty string or lowercase SHA-256")
        ids = tuple(self.candidate_ids)
        names = tuple(self.feature_names)
        features = tuple(tuple(None if value is None else _finite_tuple((value,), "candidate feature")[0]
                               for value in row) for row in self.features)
        mask = tuple(self.valid_mask)
        if not ids or any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("candidate_ids must be nonempty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("candidate_ids must be unique")
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("feature_names must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        if len(features) != len(ids) or len(mask) != len(ids):
            raise ValueError("candidate rows, features, and mask must have equal lengths")
        if any(type(value) is not bool for value in mask):
            raise ValueError("valid_mask entries must be bool")
        if any(len(row) != len(names) for row in features):
            raise ValueError("candidate feature rows must match feature_names")
        costs = self.route_costs
        if costs is not None:
            costs = tuple(None if value is None else float(value) for value in costs)
            if len(costs) != len(ids) or any(
                value is not None and not math.isfinite(value) for value in costs
            ):
                raise ValueError("route_costs must match candidates and be finite or None")
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "route_costs", costs)


@dataclass(frozen=True, slots=True)
class ScorerSnapshot:
    """Scores tied to exact observation and candidate ordering."""

    model_id: str
    snapshot_id: str
    candidate_ids: tuple[str, ...]
    scores: tuple[float, ...]
    timestamp: float
    schema_version: int = 1
    observation_digest: str = ""
    candidate_digest: str = ""

    def __post_init__(self) -> None:
        ids = tuple(self.candidate_ids)
        scores = _finite_tuple(self.scores, "scores")
        if self.schema_version not in (1, 2) or not self.model_id or not self.snapshot_id:
            raise ValueError("unsupported schema or empty scorer identity")
        digests = (self.observation_digest, self.candidate_digest)
        if self.schema_version == 2:
            if any(not isinstance(digest, str) or len(digest) != 64
                   or any(char not in "0123456789abcdef" for char in digest)
                   for digest in digests):
                raise ValueError("schema v2 scorer requires SHA-256 input digests")
        elif any(digests):
            raise ValueError("input digests require scorer schema v2")
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("scorer timestamp must be finite and nonnegative")
        if not ids or len(ids) != len(scores):
            raise ValueError("candidate_ids and scores must be nonempty and equal length")
        if len(set(ids)) != len(ids):
            raise ValueError("scorer candidate_ids must be unique")
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "scores", scores)

    def matches(self, observation: ObservationSnapshot, candidates: CandidateTable,
                *, now: float, max_age: float) -> bool:
        """Return true only for a fresh scorer result for these exact inputs."""
        if not math.isfinite(now) or not math.isfinite(max_age) or max_age < 0:
            return False
        age = now - self.timestamp
        return (self.snapshot_id == observation.snapshot_id == candidates.snapshot_id
                and self.candidate_ids == candidates.candidate_ids
                and 0.0 <= age <= max_age)


def masked_argmax(scores: Sequence[float], valid_mask: Sequence[bool],
                  *, fallback_index: int | None = None) -> int | None:
    """Choose highest finite valid score; ties resolve to lowest table index.

    If no valid finite score exists, return the explicitly designated fallback
    only when that row is valid; otherwise return ``None`` for caller fail-safe.
    """
    values = tuple(float(value) for value in scores)
    mask = tuple(valid_mask)
    if not values or len(values) != len(mask):
        raise ValueError("scores and valid_mask must be nonempty and equal length")
    if any(type(value) is not bool for value in mask):
        raise ValueError("valid_mask entries must be bool")
    eligible = [i for i, (score, valid) in enumerate(zip(values, mask))
                if valid and math.isfinite(score)]
    if eligible:
        return max(eligible, key=lambda i: (values[i], -i))
    if fallback_index is not None and 0 <= fallback_index < len(mask) and mask[fallback_index]:
        return fallback_index
    return None
