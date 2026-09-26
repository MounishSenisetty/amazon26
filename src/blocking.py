"""Candidate generation: IDF-weighted overlap of hashed name and address keys.

Every record is turned into sets of hashed keys (name-core words and word
bigrams; address words and word bigrams; postcode x name-prefix composites).
For every Source 1 record we take the top-k Source 2/3 records by IDF-weighted
cosine over shared keys, once on name keys and once on address keys, and union
the two lists. The union is exactly what the matcher scores and what
candidate_pairs.tsv reports.

To stay tractable on millions of records, a key is used for blocking only if
it occurs in at most `max_df` pool records (frequent keys such as "traders" or
a city name carry little identity). A record whose keys are all frequent keeps
its rarest key, provided it occurs in at most `fallback_df` pool records, so
common names are still blocked through their most specific key.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import normalize

from .normalize import LEGAL_TOKENS
from .parallel import SHARED, fork_map, ranges

N_FEATURES = 2 ** 25
KINDS = ("core", "core_bi", "acronym", "addr", "addr_bi", "geo", "postcode", "number", "legal")
# Upper bound on sparse similarity entries materialised per blocking task.
_COST_PER_TASK = 20_000_000


def _key_lists(name_norm, name_core, addr_norm, post, nums):
    lists = {k: [] for k in KINDS}
    first = []
    for norm, core, addr, pc, num in zip(name_norm, name_core, addr_norm, post, nums):
        ct, at, ps = core.split(), addr.split(), pc.split()
        acr = "".join(t[0] for t in ct)
        lists["core"].append(ct)
        lists["core_bi"].append([a + " " + b for a, b in zip(ct, ct[1:])])
        lists["acronym"].append([acr] if len(acr) > 1 else [])
        lists["addr"].append(at)
        lists["addr_bi"].append([a + " " + b for a, b in zip(at, at[1:])])
        lists["geo"].append(["#" + p + " " + t[:4] for p in ps for t in ct])
        lists["postcode"].append(ps)
        lists["number"].append(num.split())
        lists["legal"].append([t for t in norm.split() if t in LEGAL_TOKENS])
        first.append(ct[0] if ct else "")
    return lists, first


def _hash_chunk(bounds):
    start, end = bounds
    lists, first = _key_lists(*(SHARED[c][start:end] for c in
                                ("name_norm", "name_core", "addr_norm", "postcodes", "numbers")))
    hasher = FeatureHasher(N_FEATURES, input_type="string", alternate_sign=False, dtype=np.float32)
    mats = {}
    for kind, docs in lists.items():
        m = hasher.transform(docs).tocsr()
        m.data[:] = 1.0  # sets, not counts
        mats[kind] = m
    return mats, first


def build_keys(records, n_jobs):
    """Hash every record's keys. Returns {kind: binary CSR (records x N_FEATURES)} plus 'first'.

    'first' is an int64 hash of each record's first name-core token (0 when there is none).
    """
    cols = ("name_norm", "name_core", "addr_norm", "postcodes", "numbers")
    parts = fork_map(_hash_chunk, ranges(len(records), 100_000), n_jobs,
                     **{c: records[c].to_numpy(object) for c in cols})
    keys = {k: sp.vstack([p[0][k] for p in parts], format="csr") for k in KINDS}
    first = np.array([t for p in parts for t in p[1]], dtype=object)
    keys["first"] = np.where(first == "", 0, pd.util.hash_array(first).astype(np.int64))
    return keys


def idf_weighted(binary):
    """TF-IDF-style weights (smooth IDF over all records) with L2-normalised rows."""
    df = np.bincount(binary.indices, minlength=binary.shape[1])
    idf = (np.log((1 + binary.shape[0]) / (1 + df)) + 1).astype(np.float32)
    w = binary.astype(np.float32, copy=True)
    w.data = idf[w.indices]
    return normalize(w, norm="l2", copy=False)


def _row_ids(m):
    return np.repeat(np.arange(m.shape[0], dtype=np.int32), np.diff(m.indptr))


def _mask_entries(m, keep, remap, n_cols):
    """Keep the entries where `keep` is true and renumber their columns through `remap`."""
    cs = np.concatenate([[0], np.cumsum(keep, dtype=np.int64)])
    return sp.csr_matrix((m.data[keep], remap[m.indices[keep]], cs[m.indptr]), shape=(m.shape[0], n_cols))


def _task_bounds(cost, budget):
    """Split rows into consecutive tasks whose summed cost is about `budget` each."""
    task = (np.cumsum(cost) // budget).astype(np.int64)
    cuts = np.flatnonzero(np.diff(task)) + 1
    edges = np.concatenate([[0], cuts, [len(cost)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _topk_chunk(bounds):
    start, end = bounds
    sims = (SHARED["left"][start:end] @ SHARED["right_t"]).tocsr()
    rows = _row_ids(sims)
    order = np.lexsort((-sims.data, rows))
    rank = np.arange(len(order)) - sims.indptr[rows]  # rows is sorted, so rows[order] == rows
    sel = order[rank < SHARED["k"]]
    return rows[sel] + start, sims.indices[sel].astype(np.int32)


def channel_top_k(weights, n1, k, max_df, fallback_df, n_jobs):
    """Top-k pool rows (by pruned-key cosine) for each of the first n1 rows of `weights`.

    Returns (i, j) int32 arrays: row in Source 1, row in the pool.
    """
    left, right = weights[:n1], weights[n1:]
    if k <= 0 or n1 == 0 or right.shape[0] == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    df_pool = np.bincount(right.indices, minlength=weights.shape[1])
    entry_df = df_pool[left.indices]
    keep = (entry_df >= 1) & (entry_df <= max_df)

    # Rarest-key fallback for rows that would otherwise have no usable key.
    rows = _row_ids(left)
    has_key = np.bincount(rows[keep], minlength=n1) > 0
    present = entry_df >= 1
    rarest = np.full(n1, np.iinfo(np.int64).max)
    nonempty = np.flatnonzero(np.diff(left.indptr))
    if len(nonempty):
        vals = np.where(present, entry_df, np.iinfo(np.int64).max)
        rarest[nonempty] = np.minimum.reduceat(vals, left.indptr[nonempty])
    keep |= present & ~has_key[rows] & (entry_df == rarest[rows]) & (entry_df <= fallback_df)

    used = np.zeros(weights.shape[1], dtype=bool)
    used[left.indices[keep]] = True
    n_used = int(used.sum())
    remap = np.full(weights.shape[1], -1, dtype=np.int32)
    remap[used] = np.arange(n_used, dtype=np.int32)
    left_b = _mask_entries(left, keep, remap, n_used)
    right_t = _mask_entries(right, used[right.indices], remap, n_used).T.tocsr()

    # Split Source 1 rows into tasks of bounded similarity volume.
    cost = np.bincount(rows[keep], weights=df_pool[left.indices[keep]], minlength=n1)
    bounds = _task_bounds(cost, _COST_PER_TASK)
    parts = fork_map(_topk_chunk, bounds, n_jobs, left=left_b, right_t=right_t, k=k)
    return (np.concatenate([p[0] for p in parts]).astype(np.int32),
            np.concatenate([p[1] for p in parts]).astype(np.int32))


def generate_candidates(keys, n1, n2, k_name=15, k_addr=10, max_df=3000, fallback_df=30000, n_jobs=1, log=print):
    """Return (i, j) int32 arrays of unique candidate pairs, sorted by i then j."""
    name_w = idf_weighted(keys["core"] + keys["core_bi"])
    i1, j1 = channel_top_k(name_w, n1, k_name, max_df, fallback_df, n_jobs)
    del name_w
    log(f"blocking: name channel proposed {len(i1)} pairs")
    addr_w = idf_weighted(keys["addr"] + keys["addr_bi"] + keys["geo"])
    i2, j2 = channel_top_k(addr_w, n1, k_addr, max_df, fallback_df, n_jobs)
    del addr_w
    log(f"blocking: address channel proposed {len(i2)} pairs")
    pair_keys = np.unique(np.concatenate([i1.astype(np.int64) * n2 + j1, i2.astype(np.int64) * n2 + j2]))
    return (pair_keys // max(n2, 1)).astype(np.int32), (pair_keys % max(n2, 1)).astype(np.int32)


def candidate_lists(i, j, s1, pool):
    """Map each Source 1 entity_id to its candidate entity_ids (pool order preserved)."""
    s1_ids = s1["entity_id"].to_numpy()
    pool_ids = pool["entity_id"].to_numpy()
    lists = {sid: [] for sid in s1_ids}
    for a, b in zip(i, j):
        lists[s1_ids[a]].append(pool_ids[b])
    return lists
