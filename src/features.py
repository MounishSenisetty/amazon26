"""Pairwise features for candidate pairs, computed in bounded-memory chunks.

Pairs are (i, j) with i a Source 1 row and j a pool row. `records` stacks the
Source 1 rows and then the pool rows, so pool row j is record n1 + j; the key
matrices from blocking.build_keys use the same row order.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .blocking import idf_weighted
from .parallel import SHARED, fork_map, ranges

FEATURES = [
    "name_tfidf", "addr_tfidf",
    "name_ratio", "name_token_set", "name_partial", "core_ratio", "core_token_set",
    "core_jaro_winkler", "core_token_jaccard", "first_token_match", "acronym_match",
    "legal_form_match", "name_len_diff",
    "addr_token_set", "addr_partial", "addr_token_jaccard",
    "postcode_match", "number_jaccard",
    "same_country", "is_source3",
    "name_rank", "addr_rank", "name_gap", "addr_gap", "n_candidates", "reverse_rank",
]
NEURAL_FEATURES = FEATURES + ["emb_cos", "emb_rank"]  # with --neural: bi-encoder cosine and its rank
STAGE2_FEATURES = NEURAL_FEATURES + ["p1", "ce_score"]  # stage-1 probability and cross-encoder score
SET_FEATURES = ["core_token_jaccard", "acronym_match", "legal_form_match",
                "addr_token_jaccard", "postcode_match", "number_jaccard"]
STRING_FEATURES = [  # (feature, records column, scorer, scale)
    ("name_ratio", "name_norm", fuzz.ratio, 100),
    ("name_token_set", "name_norm", fuzz.token_set_ratio, 100),
    ("name_partial", "name_norm", fuzz.partial_ratio, 100),
    ("core_ratio", "name_core", fuzz.ratio, 100),
    ("core_token_set", "name_core", fuzz.token_set_ratio, 100),
    ("core_jaro_winkler", "name_core", JaroWinkler.similarity, 1),
    ("addr_token_set", "addr_norm", fuzz.token_set_ratio, 100),
    ("addr_partial", "addr_norm", fuzz.partial_ratio, 100),
]
_PAIRS_PER_TASK = 1_000_000

try:
    from rapidfuzz.process import cpdist
except ImportError:  # rapidfuzz < 3.6
    cpdist = None


def _dot(a, b, i, j):
    """Row-wise dot products a[i[k]] . b[j[k]]."""
    return np.asarray(a[i].multiply(b[j]).sum(axis=1), dtype=np.float32).ravel()


def _jaccard(inter, a, b):
    union = a + b - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, np.nan).astype(np.float32)


def _tri_state(inter, a, b):
    """1 if the sets share an element, 0 if both non-empty and disjoint, NaN if either is empty."""
    return np.where((a > 0) & (b > 0), (inter > 0).astype(np.float32), np.nan).astype(np.float32)


def _similarity_chunk(bounds):
    start, end = bounds
    i, j = SHARED["i"][start:end], SHARED["j"][start:end]
    return [_dot(m, m, i, j) for m in SHARED["mats"]]


def _set_chunk(bounds):
    start, end = bounds
    i, j = SHARED["i"][start:end], SHARED["j"][start:end]
    k, size = SHARED["keys"], SHARED["sizes"]

    def inter(kind):
        return _dot(k[kind], k[kind], i, j)

    acronym = (_dot(k["acronym"], k["core"], i, j) > 0) | (_dot(k["core"], k["acronym"], i, j) > 0)
    return np.column_stack([
        _jaccard(inter("core"), size["core"][i], size["core"][j]),
        acronym.astype(np.float32),
        _tri_state(inter("legal"), size["legal"][i], size["legal"][j]),
        _jaccard(inter("addr"), size["addr"][i], size["addr"][j]),
        _tri_state(inter("postcode"), size["postcode"][i], size["postcode"][j]),
        _jaccard(inter("number"), size["number"][i], size["number"][j]),
    ])


def similarity_features(i, j, n1, keys, n_jobs):
    """IDF-weighted cosine on name keys and on address keys, for every pair."""
    mats = [idf_weighted(keys["core"] + keys["core_bi"]), idf_weighted(keys["addr"] + keys["addr_bi"])]
    parts = fork_map(_similarity_chunk, ranges(len(i), _PAIRS_PER_TASK), n_jobs,
                     i=i, j=j.astype(np.int64) + n1, mats=mats)
    return {name: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0, np.float32)
            for k, name in enumerate(["name_tfidf", "addr_tfidf"])}


def context_features(pairs):
    """Rank/gap features within each Source 1 entity's list and across each pool record's rivals."""
    by_s1 = pairs.groupby("i", sort=False)
    out = {
        "name_rank": by_s1["name_tfidf"].rank(ascending=False, method="min"),
        "addr_rank": by_s1["addr_tfidf"].rank(ascending=False, method="min"),
        "name_gap": by_s1["name_tfidf"].transform("max") - pairs["name_tfidf"],
        "addr_gap": by_s1["addr_tfidf"].transform("max") - pairs["addr_tfidf"],
        "n_candidates": by_s1["j"].transform("size"),
    }
    if "emb_cos" in pairs:
        out["emb_rank"] = by_s1["emb_cos"].rank(ascending=False, method="min")
    combined = pairs["name_tfidf"] + pairs["addr_tfidf"]
    out["reverse_rank"] = combined.groupby(pairs["j"], sort=False).rank(ascending=False, method="min")
    return {k: v.to_numpy(np.float32) for k, v in out.items()}


def _string_scores(scorer, left, right, n_jobs):
    if cpdist is not None:
        return cpdist(left, right, scorer=scorer, workers=n_jobs, dtype=np.float32)
    return np.fromiter((scorer(a, b) for a, b in zip(left, right)), dtype=np.float32, count=len(left))


def pair_features(pairs, records, keys, n1, n_jobs):
    """Add the string, set and record-level features to `pairs` (which has i, j and the context features)."""
    i = pairs["i"].to_numpy(np.int64)
    j = pairs["j"].to_numpy(np.int64) + n1
    n = len(i)

    cols = {name: np.empty(n, np.float32) for name, *_ in STRING_FEATURES}
    text = {c: records[c].to_numpy(object) for c in ("name_norm", "name_core", "addr_norm")}
    for start, end in ranges(n, _PAIRS_PER_TASK):
        li, rj = i[start:end], j[start:end]
        for name, col, scorer, scale in STRING_FEATURES:
            cols[name][start:end] = _string_scores(scorer, text[col][li], text[col][rj], n_jobs) / scale

    sizes = {kind: np.diff(keys[kind].indptr) for kind in ("core", "legal", "addr", "postcode", "number")}
    parts = fork_map(_set_chunk, ranges(n, _PAIRS_PER_TASK), n_jobs, i=i, j=j, keys=keys, sizes=sizes)
    set_arr = np.concatenate(parts) if parts else np.zeros((0, len(SET_FEATURES)), np.float32)
    for k, name in enumerate(SET_FEATURES):
        cols[name] = set_arr[:, k]

    first = keys["first"]
    cols["first_token_match"] = ((first[i] == first[j]) & (first[i] != 0)).astype(np.float32)
    core_len = records["name_core"].str.len().to_numpy(np.float32)
    lc, rc = core_len[i], core_len[j]
    cols["name_len_diff"] = np.abs(lc - rc) / np.maximum(np.maximum(lc, rc), 1)
    country = pd.factorize(records["country_norm"])[0]
    cols["same_country"] = (country[i] == country[j]).astype(np.float32)
    is_s3 = records["entity_id"].str.startswith("S3-").to_numpy(bool)
    cols["is_source3"] = is_s3[j].astype(np.float32)
    for name, values in cols.items():
        pairs[name] = values
    return pairs
