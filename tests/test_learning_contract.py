from dataclasses import FrozenInstanceError

import pytest

from frc_defense.learning_contract import (
    CandidateTable,
    ObservationSnapshot,
    ScorerSnapshot,
    masked_argmax,
)


def observation(snapshot_id="obs-1"):
    return ObservationSnapshot(snapshot_id, "episode-1", 4, 10.0,
                               (1, 1, 2), (0.0, 1.0), (0.0,) * 7)


def candidates(snapshot_id="obs-1"):
    return CandidateTable(snapshot_id, ("safe", "score", "ferry"),
                          ((1.,), (2.,), (3.,)), (True, True, False),
                          (0., 2., None))


def test_observation_schema_is_validated_and_immutable():
    snap = observation()
    with pytest.raises(FrozenInstanceError):
        snap.snapshot_id = "changed"
    with pytest.raises(ValueError, match="seven"):
        ObservationSnapshot("o", "e", 0, 0., (1, 1, 1), (0.,), (0.,))
    with pytest.raises(ValueError, match="expected 2"):
        ObservationSnapshot("o", "e", 0, 0., (1, 1, 2), (0.,), (0.,) * 7)


def test_candidate_table_rejects_misaligned_or_invalid_masks():
    with pytest.raises(ValueError, match="equal lengths"):
        CandidateTable("o", ("a", "b"), ((0.,),), (True, True))
    with pytest.raises(ValueError, match="must be bool"):
        CandidateTable("o", ("a",), ((0.,),), (1,))
    with pytest.raises(ValueError, match="unique"):
        CandidateTable("o", ("a", "a"), ((0.,), (1.,)), (True, True))


def test_scorer_requires_exact_snapshot_order_and_freshness():
    obs, table = observation(), candidates()
    scorer = ScorerSnapshot("model-v1", "obs-1", table.candidate_ids,
                            (0.2, 0.9, 99.), 10.5)
    assert scorer.matches(obs, table, now=11., max_age=1.)
    assert not scorer.matches(obs, table, now=11.6, max_age=1.)
    assert not scorer.matches(observation("other"), table, now=11., max_age=1.)
    reversed_table = CandidateTable("obs-1", tuple(reversed(table.candidate_ids)),
                                    tuple(reversed(table.features)),
                                    tuple(reversed(table.valid_mask)))
    assert not scorer.matches(obs, reversed_table, now=11., max_age=1.)
    assert not scorer.matches(obs, table, now=9., max_age=1.)


def test_masked_argmax_masks_invalid_rows_and_resolves_ties_by_order():
    assert masked_argmax((8., 4., 8.), (False, True, True)) == 2
    assert masked_argmax((2., 2., -1.), (True, True, True)) == 0


def test_masked_argmax_uses_only_valid_fallback_when_all_scores_unusable():
    assert masked_argmax((float("nan"), float("inf")), (True, True),
                         fallback_index=1) == 1
    assert masked_argmax((1., 2.), (False, False), fallback_index=0) is None
    assert masked_argmax((1.,), (True,), fallback_index=3) == 0
