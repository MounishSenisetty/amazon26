"""Pieces added for million-record scale: vectorised scoring, streaming writer, pruned blocking."""

import numpy as np
import pandas as pd
import pytest

from src.blocking import build_keys, generate_candidates
from src.metrics import macro_from_counts, macro_scores
from src.normalize import prepare
from src.tsv import MATCH_HEADER, write_pair_lists


def test_macro_from_counts_matches_macro_scores():
    gold = {"a": [], "b": ["x"], "c": ["x", "y"], "d": ["z"], "e": []}
    preds = {"a": [], "b": ["x", "q"], "c": ["y"], "d": ["w"], "e": ["v"]}
    ids = list(gold)
    pred_n = [len(preds[e]) for e in ids]
    tp = [len(set(preds[e]) & set(gold[e])) for e in ids]
    gold_n = [len(gold[e]) for e in ids]
    fast, slow = macro_from_counts(pred_n, tp, gold_n), macro_scores(preds, gold, ids)
    for key in ("f05", "precision", "recall"):
        assert fast[key] == pytest.approx(slow[key])


def test_write_pair_lists_groups_by_entity_in_file_order(tmp_path):
    path = tmp_path / "m.tsv"
    write_pair_lists(path, MATCH_HEADER, ["S1-1", "S1-2", "S1-3"],
                     np.array([2, 0, 2]), np.array([1, 0, 0]), np.array(["S2-a", "S3-b"], dtype=object))
    assert path.read_text() == ("source1_entity_id\tmatched_entity_ids\n"
                                "S1-1\tS2-a\nS1-2\t\nS1-3\tS3-b,S2-a\n")


def _records(names, addresses):
    df = pd.DataFrame({"entity_id": [f"E{k}" for k in range(len(names))], "business_name": names,
                       "business_address": addresses, "country": "US"})
    return prepare(df)


def test_blocking_finds_match_despite_frequent_keys():
    # Source 1 row 0 has only frequent keys ("common", "shop", street, city), but its postcode composite is rare.
    s1_names, s1_addr = ["Common Shop", "Rare Widget"], ["1 Main St, Springfield 12345", "9 Elm St, Shelbyville 67890"]
    pool_names = ["Common Shop"] + ["Common Shop"] * 30 + ["Rare Widgit"]
    pool_addr = ["1 Main St, Springfield 12345"] + ["5 Main St, Springfield 99999"] * 30 + ["9 Elm Street, 67890"]
    records = _records(s1_names + pool_names, s1_addr + pool_addr)
    keys = build_keys(records, n_jobs=1)
    i, j = generate_candidates(keys, 2, len(pool_names), k_name=1, k_addr=1, max_df=5, fallback_df=10, n_jobs=1,
                               log=lambda _: None)
    pairs = set(zip(i.tolist(), j.tolist()))
    assert (0, 0) in pairs  # found through the postcode x name-prefix composite
    assert (1, len(pool_names) - 1) in pairs  # typo in the name, found through the address


def test_xgboost_classifier_falls_back_to_cpu_and_scores():
    pytest.importorskip("xgboost")
    from src.features import FEATURES
    from src.matcher import fit, score

    rng = np.random.default_rng(0)
    pairs = pd.DataFrame(rng.random((400, len(FEATURES)), dtype=np.float32), columns=FEATURES)
    labels = (pairs["name_tfidf"] > 0.5).astype(np.int8).to_numpy()
    model = fit(pairs, labels, seed=0, model="xgboost", device="cuda", log=lambda _: None)
    prob = score(model, pairs)
    assert prob.shape == (400,) and ((prob > 0.5) == labels).mean() > 0.9
