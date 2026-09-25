"""Small synthetic dataset in the challenge layout, used only by the tests.

It imitates the documented noise patterns (legal-suffix drift, abbreviations,
typos, missing postcodes, reordered tokens) and, like the real test set, puts
a France slice in test only.
"""

import random
from pathlib import Path

WORDS = {
    "US": ["Summit", "Harbor", "Pioneer", "Liberty", "Cedar", "Evergreen", "Granite", "Maple",
           "Redwood", "Silver", "Eagle", "Prairie", "Beacon", "Hudson", "Frontier", "Atlas"],
    "India": ["Sharma", "Ganesh", "Lakshmi", "Shree", "Krishna", "Balaji", "Om", "Sai",
              "Annapurna", "Vinayak", "Surya", "Tulsi", "Kaveri", "Ashoka", "Mahalaxmi", "Nandi"],
    "France": ["Lumiere", "Belleville", "Saint", "Provence", "Marais", "Etoile", "Rivoli",
               "Montmartre", "Bastille", "Garonne", "Loire", "Opera", "Chateau", "Vendome"],
}
TRADES = {
    "US": ["Hardware", "Dental", "Logistics", "Bakery", "Motors", "Consulting", "Pharmacy"],
    "India": ["Traders", "Textiles", "Sweets", "Electricals", "Jewellers", "Enterprises", "Agencies"],
    "France": ["Boulangerie", "Conseil", "Transports", "Pharmacie", "Immobilier", "Optique"],
}
LEGAL = {
    "US": [("Corporation", "Corp"), ("Incorporated", "Inc"), ("LLC", "LLC"), ("Company", "Co")],
    "India": [("Private Limited", "Pvt Ltd"), ("Limited", "Ltd"), ("", "")],
    "France": [("SARL", "SARL"), ("SAS", "SAS"), ("", "")],
}
STREETS = {
    "US": [("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Boulevard", "Blvd")],
    "India": [("Road", "Rd"), ("Main Road", "Main Rd"), ("Nagar", "Ngr"), ("Marg", "Marg")],
    "France": [("Rue", "Rue"), ("Avenue", "Av"), ("Boulevard", "Bd")],
}
CITIES = {
    "US": [("Austin", "TX"), ("Denver", "CO"), ("Seattle", "WA"), ("Boston", "MA")],
    "India": [("Bengaluru", "Karnataka"), ("Pune", "Maharashtra"), ("Chennai", "Tamil Nadu")],
    "France": [("Lyon", ""), ("Paris", ""), ("Bordeaux", "")],
}


def _typo(rng, text):
    if len(text) < 5:
        return text
    k = rng.randrange(1, len(text) - 1)
    return text[:k] + text[k + 1] + text[k] + text[k + 2:]


def _business(rng, country):
    w = rng.sample(WORDS[country], 2)
    legal = rng.choice(LEGAL[country])
    trade = rng.choice(TRADES[country])
    street_word = rng.choice(WORDS[country])
    street = rng.choice(STREETS[country])
    city, region = rng.choice(CITIES[country])
    return {
        "core": f"{w[0]} {w[1]} {trade}",
        "legal": legal,
        "number": str(rng.randint(1, 999)),
        "street_word": street_word,
        "street": street,
        "city": city,
        "region": region,
        "post": str(rng.randint(100000, 999999) if country == "India" else rng.randint(10000, 99999)),
        "country": country,
    }


def _render(rng, b, noisy):
    long_legal, short_legal = b["legal"]
    legal = rng.choice([long_legal, short_legal]) if noisy else long_legal
    core = b["core"]
    if noisy and rng.random() < 0.3:
        core = _typo(rng, core)
    if noisy and rng.random() < 0.15:
        a, rest = core.split(" ", 1)
        core = f"{rest} {a}"
    name = f"{core} {legal}".strip()
    long_st, short_st = b["street"]
    street = rng.choice([long_st, short_st]) if noisy else long_st
    parts = [f"{b['number']} {b['street_word']} {street}", b["city"]]
    if b["region"] and not (noisy and rng.random() < 0.3):
        parts.append(b["region"])
    if not (noisy and rng.random() < 0.3):
        parts.append(b["post"])
    if noisy and rng.random() < 0.1:
        parts.insert(0, "Near Bus Stand")
    return name, ", ".join(parts)


def _write(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


def _make_split(rng, split, countries, n_per_country):
    s1, s2, s3, gt = [], [], [], []
    counters = {1: 0, 2: 0, 3: 0}

    def new_id(src):
        counters[src] += 1
        return f"S{src}-{counters[src]:05d}"

    for country in countries:
        for _ in range(n_per_country):
            b = _business(rng, country)
            sid = new_id(1)
            name, addr = _render(rng, b, noisy=False)
            s1.append((sid, name, addr, country))
            matches = []
            for _ in range(rng.choice([0, 0, 1, 1, 1, 2, 3])):
                src = rng.choice([2, 3])
                pid = new_id(src)
                n, a = _render(rng, b, noisy=True)
                (s2 if src == 2 else s3).append((pid, n, a, country))
                matches.append(pid)
            gt.append((sid, ",".join(matches)))
        for _ in range(n_per_country // 2):  # pool records with no Source 1 counterpart
            b = _business(rng, country)
            src = rng.choice([2, 3])
            n, a = _render(rng, b, noisy=True)
            (s2 if src == 2 else s3).append((new_id(src), n, a, country))
    rng.shuffle(s2)
    rng.shuffle(s3)
    return s1, s2, s3, gt


def make_dataset(root, seed=0, n_per_country=60):
    """Write dataset/train (US, India) and dataset/test (US, India, France) under root."""
    rng = random.Random(seed)
    root = Path(root)
    header = ["entity_id", "business_name", "business_address", "country"]
    for split, countries in (("train", ["US", "India"]), ("test", ["US", "India", "France"])):
        s1, s2, s3, gt = _make_split(rng, split, countries, n_per_country)
        for n, rows in ((1, s1), (2, s2), (3, s3)):
            _write(root / split / f"{split}_source{n}.tsv", rows, header)
        if split == "train":
            _write(root / split / "train_ground_truth.tsv", gt, ["source1_entity_id", "matched_entity_ids"])
        else:
            _write(root / "test_ground_truth.tsv", gt, ["source1_entity_id", "matched_entity_ids"])
    return root
