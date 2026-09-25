"""Pairwise features for candidate pairs."""

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .normalize import LEGAL_TOKENS

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


def _jaccard(a, b):
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


def _acronym(tokens):
    return "".join(t[0] for t in tokens if t)


def _tri_state(a, b):
    """1 if the sets share an element, 0 if both non-empty and disjoint, NaN if either is empty."""
    if not a or not b:
        return np.nan
    return float(bool(a & b))


def build_features(cand, s1, pool):
    """Return a copy of cand with every column in FEATURES added."""
    out = cand.copy()
    left = s1.iloc[out["i"].to_numpy()].reset_index(drop=True)
    right = pool.iloc[out["j"].to_numpy()].reset_index(drop=True)

    rows = []
    for (ln, lc, la, lp, lnum, lcty), (rn, rc, ra, rp, rnum, rcty, rid) in zip(
        left[["name_norm", "name_core", "addr_norm", "postcodes", "numbers", "country_norm"]].itertuples(index=False),
        right[["name_norm", "name_core", "addr_norm", "postcodes", "numbers", "country_norm", "entity_id"]].itertuples(index=False),
    ):
        lct, rct = lc.split(), rc.split()
        l_legal = {t for t in ln.split() if t in LEGAL_TOKENS}
        r_legal = {t for t in rn.split() if t in LEGAL_TOKENS}
        la_set, ra_set = set(la.split()), set(ra.split())
        l_acr, r_acr = _acronym(lct), _acronym(rct)
        rows.append((
            fuzz.ratio(ln, rn) / 100,
            fuzz.token_set_ratio(ln, rn) / 100,
            fuzz.partial_ratio(ln, rn) / 100,
            fuzz.ratio(lc, rc) / 100,
            fuzz.token_set_ratio(lc, rc) / 100,
            JaroWinkler.similarity(lc, rc),
            _jaccard(set(lct), set(rct)),
            float(bool(lct and rct and lct[0] == rct[0])),
            float((len(l_acr) > 1 and l_acr in rct) or (len(r_acr) > 1 and r_acr in lct)),
            _tri_state(l_legal, r_legal),
            abs(len(lc) - len(rc)) / max(len(lc), len(rc), 1),
            fuzz.token_set_ratio(la, ra) / 100,
            fuzz.partial_ratio(la, ra) / 100,
            _jaccard(la_set, ra_set),
            _tri_state(lp, rp),
            _jaccard(lnum, rnum),
            float(lcty == rcty),
            float(rid.startswith("S3-")),
        ))
    cols = FEATURES[2:20]
    arr = np.array(rows, dtype=np.float32).reshape(len(rows), len(cols))
    for k, col in enumerate(cols):
        out[col] = arr[:, k]

    # Context within each Source 1 entity's candidate list, and across the pool record's rivals.
    by_s1 = out.groupby("i")
    out["name_rank"] = by_s1["name_tfidf"].rank(ascending=False, method="min")
    out["addr_rank"] = by_s1["addr_tfidf"].rank(ascending=False, method="min")
    out["name_gap"] = by_s1["name_tfidf"].transform("max") - out["name_tfidf"]
    out["addr_gap"] = by_s1["addr_tfidf"].transform("max") - out["addr_tfidf"]
    out["n_candidates"] = by_s1["j"].transform("size")
    combined = out["name_tfidf"] + out["addr_tfidf"]
    out["reverse_rank"] = combined.groupby(out["j"]).rank(ascending=False, method="min")
    return out
