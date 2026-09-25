"""Candidate generation: TF-IDF nearest neighbours on names and on addresses.

For every Source 1 record we take the top-k Source 2/3 records by character
n-gram cosine on the normalised name and, separately, on the normalised
address, and union the two lists. The union is exactly what the matcher scores
and what candidate_pairs.tsv reports.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Upper bound on dense similarity cells materialised per chunk (~200 MB as float32).
_CELLS_PER_CHUNK = 50_000_000


def _vectorize(left, right):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32)
    vec.fit(pd.concat([left, right], ignore_index=True))
    return vec.transform(left), vec.transform(right)


def _top_k(left, right, k):
    """Yield (left_row, right_row, cosine) for each left row's k most similar right rows."""
    k = min(k, right.shape[0])
    if k == 0:
        return
    chunk = max(1, _CELLS_PER_CHUNK // max(1, right.shape[0]))
    right_t = right.T.tocsc()
    for start in range(0, left.shape[0], chunk):
        sims = (left[start:start + chunk] @ right_t).toarray()
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(sims.shape[0]), k)
        cols = idx.ravel()
        vals = sims[rows, cols]
        keep = vals > 0
        yield from zip(rows[keep] + start, cols[keep], vals[keep])


def generate_candidates(s1, pool, k_name=15, k_addr=10):
    """Return a DataFrame of candidate pairs with blocking-stage similarities.

    Columns: i (row in s1), j (row in pool), name_tfidf, addr_tfidf.
    Both similarities are filled for every pair, whichever channel proposed it.
    """
    name_l, name_r = _vectorize(s1["name_norm"], pool["name_norm"])
    addr_l, addr_r = _vectorize(s1["addr_norm"], pool["addr_norm"])

    pairs = set()
    for i, j, _ in _top_k(name_l, name_r, k_name):
        pairs.add((int(i), int(j)))
    for i, j, _ in _top_k(addr_l, addr_r, k_addr):
        pairs.add((int(i), int(j)))

    cand = pd.DataFrame(sorted(pairs), columns=["i", "j"], dtype=np.int64)
    if cand.empty:
        cand["name_tfidf"] = cand["addr_tfidf"] = np.float32()
        return cand
    i, j = cand["i"].to_numpy(), cand["j"].to_numpy()
    cand["name_tfidf"] = np.asarray(name_l[i].multiply(name_r[j]).sum(axis=1)).ravel()
    cand["addr_tfidf"] = np.asarray(addr_l[i].multiply(addr_r[j]).sum(axis=1)).ravel()
    return cand


def candidate_lists(cand, s1, pool):
    """Map each Source 1 entity_id to its candidate entity_ids (pool order preserved)."""
    s1_ids = s1["entity_id"].to_numpy()
    pool_ids = pool["entity_id"].to_numpy()
    lists = {sid: [] for sid in s1_ids}
    for i, j in zip(cand["i"].to_numpy(), cand["j"].to_numpy()):
        lists[s1_ids[i]].append(pool_ids[j])
    return lists
