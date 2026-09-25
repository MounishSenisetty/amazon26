import pytest

from src.metrics import blocking_scores, entity_f05, macro_scores


def test_worked_example_from_challenge():
    pred = ["S2-00047", "S2-00193", "S3-00812"]
    gold = ["S2-00047", "S3-00812"]
    assert entity_f05(pred, gold) == pytest.approx(0.714, abs=1e-3)


@pytest.mark.parametrize("pred, gold, expected", [
    ([], [], 1.0),                    # correct singleton
    (["S2-00001"], [], 0.0),          # false merge on a singleton
    ([], ["S2-00001"], 0.0),          # missed every match
    (["S2-00001"], ["S2-00001"], 1.0),
])
def test_singleton_truth_table(pred, gold, expected):
    assert entity_f05(pred, gold) == expected


def test_macro_average_counts_missing_predictions_as_empty():
    gold = {"S1-1": [], "S1-2": ["S2-1"]}
    scores = macro_scores({}, gold, ["S1-1", "S1-2"])
    assert scores["f05"] == 0.5


def test_blocking_scores():
    gold = {"S1-1": ["S2-1", "S3-1"], "S1-2": []}
    cands = {"S1-1": ["S2-1", "S2-9"], "S1-2": ["S3-4"]}
    s = blocking_scores(cands, gold, ["S1-1", "S1-2"], pool_size=10)
    assert s["pair_completeness"] == 0.5
    assert s["reduction_ratio"] == pytest.approx(1 - 3 / 20)
