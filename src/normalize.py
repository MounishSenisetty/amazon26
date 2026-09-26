"""Text normalisation for business names and addresses.

Nothing here is keyed on the country label, so records from countries unseen
in training (e.g. France) go through exactly the same path.
"""

import itertools
import re
import unicodedata

import numpy as np
import pandas as pd

from .parallel import SHARED, fork_map, ranges

# Legal-form variants mapped to one canonical token each.
LEGAL_FORMS = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "limited": "ltd", "ltd": "ltd",
    "corporation": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc",
    "company": "co", "co": "co", "cie": "co",
    "llc": "llc", "llp": "llp", "plc": "plc", "lp": "lp",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv",
}
LEGAL_TOKENS = set(LEGAL_FORMS.values())
NAME_STOPWORDS = {"the", "and", "of", "dba", "et", "le", "la", "les", "de", "du", "des"}

ADDRESS_ABBREVIATIONS = {
    "road": "rd", "street": "st", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "bd": "blvd", "drive": "dr", "lane": "ln",
    "court": "ct", "place": "pl", "square": "sq", "highway": "hwy",
    "parkway": "pkwy", "suite": "ste", "floor": "fl", "building": "bldg",
    "apartment": "apt", "number": "no", "north": "n", "south": "s",
    "east": "e", "west": "w", "sector": "sec", "opposite": "opp",
    "nagar": "ngr", "saint": "st",
}

_POSTCODE = re.compile(r"\b\d{5,6}\b")
_NUMBER = re.compile(r"\d+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def fold(text):
    """Lowercase, strip accents, turn '&' into 'and', drop apostrophes, keep [a-z0-9 ]."""
    text = text or ""
    if not text.isascii():  # NFKD is the identity on ASCII, so most records skip it
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ").replace("'", "").replace("’", "")
    return " ".join(_NON_ALNUM.sub(" ", text).split())


def name_tokens(name):
    return [LEGAL_FORMS.get(t, t) for t in fold(name).split()]


def normalize_name(name):
    """Full normalised name, legal forms canonicalised."""
    return " ".join(name_tokens(name))


def name_core(name):
    """Normalised name without legal forms or stopwords; falls back to the full name."""
    tokens = [t for t in name_tokens(name) if t not in LEGAL_TOKENS and t not in NAME_STOPWORDS]
    return " ".join(tokens) or normalize_name(name)


def normalize_address(address):
    return " ".join(ADDRESS_ABBREVIATIONS.get(t, t) for t in fold(address).split())


def postcodes(address):
    """5-digit (US ZIP, French code postal) and 6-digit (Indian PIN) codes in the address."""
    return frozenset(_POSTCODE.findall(address or ""))


def numbers(address):
    return frozenset(_NUMBER.findall(address or ""))


def _prepare_chunk(bounds):
    start, end = bounds
    cols = ([], [], [], [], [])
    for name, address in zip(SHARED["names"][start:end], SHARED["addresses"][start:end]):
        tokens = name_tokens(name)
        norm = " ".join(tokens)
        cols[0].append(norm)
        cols[1].append(" ".join(t for t in tokens if t not in LEGAL_TOKENS and t not in NAME_STOPWORDS) or norm)
        cols[2].append(normalize_address(address))
        cols[3].append(" ".join(sorted(postcodes(address))))
        cols[4].append(" ".join(sorted(numbers(address))))
    return cols


def prepare(df, n_jobs=1):
    """Return the normalised columns used by blocking and features.

    Columns: entity_id, country, country_norm, name_norm, name_core, addr_norm,
    postcodes and numbers (the last two as space-separated sorted strings).
    The raw name and address are not kept, to save memory at scale.
    """
    tasks = ranges(len(df), 100_000)
    parts = fork_map(_prepare_chunk, tasks, n_jobs,
                     names=df["business_name"].to_numpy(object), addresses=df["business_address"].to_numpy(object))
    country = pd.Categorical(df["country"].to_numpy(object))
    out = pd.DataFrame({"entity_id": df["entity_id"].to_numpy(object), "country": country})
    folded = np.array([fold(c) for c in country.categories], dtype=object)
    out["country_norm"] = pd.Categorical(folded[country.codes])  # labels such as "US" and "us" may merge
    for k, col in enumerate(["name_norm", "name_core", "addr_norm", "postcodes", "numbers"]):
        out[col] = np.array(list(itertools.chain.from_iterable(p[k] for p in parts)), dtype=object)
    return out
