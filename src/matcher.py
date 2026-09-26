"""Pair classifier, F0.5 threshold tuning and final match selection."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FEATURES
from .metrics import macro_from_counts

_SCORE_CHUNK = 2_000_000


def fit(pairs, labels, seed):
    """Gradient-boosted trees trained from scratch on our features (no pretrained weights)."""
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed,
    )
    model.fit(pairs[FEATURES].to_numpy(np.float32), labels)
    return model


def score(model, pairs):
    prob = np.zeros(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), _SCORE_CHUNK):  # chunked to avoid one huge float copy
        chunk = pairs.iloc[start:start + _SCORE_CHUNK]
        prob[start:start + len(chunk)] = model.predict_proba(chunk[FEATURES].to_numpy(np.float32))[:, 1]
    return prob


def best_owner(j, prob):
    """Mask of the highest-probability pair for each pool record j."""
    order = np.lexsort((-prob, j))
    first = np.ones(len(order), dtype=bool)
    first[1:] = j[order][1:] != j[order][:-1]
    best = np.zeros(len(prob), dtype=bool)
    best[order[first]] = True
    return best


def select(pairs, prob, threshold, exclusive):
    """Return a boolean mask of accepted pairs.

    With exclusive=True a pool record is given only to the Source 1 entity it
    scores highest with, which is correct when no pool record is shared
    between two Source 1 entities in the ground truth.
    """
    keep = prob >= threshold
    if exclusive and len(prob):
        keep &= best_owner(pairs["j"].to_numpy(), prob)
    return keep


def match_lists(pairs, mask, s1, pool):
    s1_ids = s1["entity_id"].to_numpy()
    pool_ids = pool["entity_id"].to_numpy()
    lists = {sid: [] for sid in s1_ids}
    for i, j in zip(pairs["i"].to_numpy()[mask], pairs["j"].to_numpy()[mask]):
        lists[s1_ids[i]].append(pool_ids[j])
    return lists


def tune_threshold(entity_pos, j, prob, labels, gold_n, exclusive):
    """Pick the threshold that maximises macro F0.5 (singletons included).

    entity_pos maps each pair to its entity's position in gold_n (the gold
    list size of every evaluated entity, including entities with no pairs).
    """
    best_mask = best_owner(j, prob) if exclusive and len(prob) else np.ones(len(prob), dtype=bool)
    labels = labels.astype(bool)
    best_t, best = 0.5, None
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        keep = (prob >= t) & best_mask
        pred_n = np.bincount(entity_pos[keep], minlength=len(gold_n))
        tp = np.bincount(entity_pos[keep & labels], minlength=len(gold_n))
        scores = macro_from_counts(pred_n, tp, gold_n)
        if best is None or scores["f05"] >= best["f05"]:  # ties go to the higher, more precise threshold
            best_t, best = float(t), scores
    return best_t, best
