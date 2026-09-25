"""Pair classifier, F0.5 threshold tuning and final match selection."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FEATURES
from .metrics import macro_scores


def fit(pairs, labels, seed):
    """Gradient-boosted trees trained from scratch on our features (no pretrained weights)."""
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed,
    )
    model.fit(pairs[FEATURES].to_numpy(np.float32), labels)
    return model


def score(model, pairs):
    if pairs.empty:
        return np.zeros(0)
    return model.predict_proba(pairs[FEATURES].to_numpy(np.float32))[:, 1]


def select(pairs, prob, threshold, exclusive):
    """Return a boolean mask of accepted pairs.

    With exclusive=True a pool record is given only to the Source 1 entity it
    scores highest with, which is correct when no pool record is shared
    between two Source 1 entities in the ground truth.
    """
    keep = prob >= threshold
    if exclusive and len(prob):
        order = np.lexsort((-prob, pairs["j"].to_numpy()))
        first = np.ones(len(order), dtype=bool)
        first[1:] = pairs["j"].to_numpy()[order][1:] != pairs["j"].to_numpy()[order][:-1]
        best = np.zeros(len(prob), dtype=bool)
        best[order[first]] = True
        keep &= best
    return keep


def match_lists(pairs, mask, s1, pool):
    s1_ids = s1["entity_id"].to_numpy()
    pool_ids = pool["entity_id"].to_numpy()
    lists = {sid: [] for sid in s1_ids}
    for i, j in zip(pairs["i"].to_numpy()[mask], pairs["j"].to_numpy()[mask]):
        lists[s1_ids[i]].append(pool_ids[j])
    return lists


def tune_threshold(pairs, prob, s1, pool, gold, entity_ids, exclusive):
    """Pick the threshold that maximises macro F0.5 over entity_ids (singletons included)."""
    best_t, best = 0.5, None
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        preds = match_lists(pairs, select(pairs, prob, t, exclusive), s1, pool)
        scores = macro_scores(preds, gold, entity_ids)
        if best is None or scores["f05"] >= best["f05"]:  # ties go to the higher, more precise threshold
            best_t, best = float(t), scores
    return best_t, best
