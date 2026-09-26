"""Challenge metric (macro F0.5 over all Source 1 entities) and blocking metrics."""

import numpy as np

BETA_SQ = 0.25


def entity_f05(predicted, gold):
    """F0.5 for one Source 1 entity. Empty vs empty scores 1.0; any other miss scores 0.0."""
    predicted, gold = set(predicted), set(gold)
    if not predicted and not gold:
        return 1.0
    tp = len(predicted & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(predicted), tp / len(gold)
    return (1 + BETA_SQ) * p * r / (BETA_SQ * p + r)


def macro_scores(predictions, gold, entity_ids):
    """Macro precision, recall and F0.5 over entity_ids (missing predictions count as empty).

    Per-entity precision is 1.0 for an empty prediction and recall is 1.0 for an
    empty gold list, so singletons do not drag the P/R diagnostics to zero.
    """
    f, p, r = [], [], []
    for eid in entity_ids:
        pred, true = set(predictions.get(eid, [])), set(gold.get(eid, []))
        tp = len(pred & true)
        f.append(entity_f05(pred, true))
        p.append(tp / len(pred) if pred else float(not true))
        r.append(tp / len(true) if true else 1.0)
    n = max(len(f), 1)
    return {"f05": sum(f) / n, "precision": sum(p) / n, "recall": sum(r) / n, "entities": len(f)}


def macro_from_counts(pred_n, tp, gold_n):
    """Vectorised macro_scores from per-entity counts: |predicted|, |predicted & gold|, |gold|."""
    pred_n, tp, gold_n = (np.asarray(a, dtype=np.float64) for a in (pred_n, tp, gold_n))
    with np.errstate(invalid="ignore", divide="ignore"):
        p = tp / pred_n
        r = tp / gold_n
        f = np.where(tp > 0, (1 + BETA_SQ) * p * r / (BETA_SQ * p + r), 0.0)
    f = np.where((pred_n == 0) & (gold_n == 0), 1.0, f)
    precision = np.where(pred_n > 0, p, (gold_n == 0).astype(np.float64))
    recall = np.where(gold_n > 0, r, 1.0)
    n = max(len(f), 1)
    return {"f05": float(f.sum() / n), "precision": float(precision.sum() / n),
            "recall": float(recall.sum() / n), "entities": len(f)}


def blocking_scores(candidates, gold, entity_ids, pool_size):
    """Pair completeness (recall ceiling), reduction ratio and candidate volume."""
    true_pairs = found = n_cand = 0
    for eid in entity_ids:
        cands, true = set(candidates.get(eid, [])), set(gold.get(eid, []))
        true_pairs += len(true)
        found += len(cands & true)
        n_cand += len(cands)
    total = max(len(entity_ids) * pool_size, 1)
    return {
        "pair_completeness": found / true_pairs if true_pairs else 1.0,
        "reduction_ratio": 1 - n_cand / total,
        "candidate_pairs": n_cand,
        "mean_candidates_per_entity": n_cand / max(len(entity_ids), 1),
    }
