"""Text normalisation for business names and addresses.

Nothing here is keyed on the country label, so records from countries unseen
in training (e.g. France) go through exactly the same path.
"""

import re
import unicodedata

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


def fold(text):
    """Lowercase, strip accents, turn '&' into 'and', drop apostrophes, keep [a-z0-9 ]."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = text.replace("&", " and ").replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


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


def prepare(df):
    """Add normalised columns used by blocking and features."""
    out = df.copy()
    out["name_norm"] = out["business_name"].map(normalize_name)
    out["name_core"] = out["business_name"].map(name_core)
    out["addr_norm"] = out["business_address"].map(normalize_address)
    out["postcodes"] = out["business_address"].map(postcodes)
    out["numbers"] = out["business_address"].map(numbers)
    out["country_norm"] = out["country"].map(fold)
    return out
