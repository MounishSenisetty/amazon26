"""Reading challenge TSVs and writing the two submission files."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


def read_tsv(path):
    """Read a challenge TSV with every cell as a string and empty cells kept as ''.

    keep_default_na=False stops pandas turning empty ID lists (and names such
    as "NA") into NaN; QUOTE_NONE keeps stray double quotes in names literal.
    """
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE
    )


def read_sources(split_dir, split):
    """Return (source1, pool) where pool stacks Source 2 and Source 3 records."""
    split_dir = Path(split_dir)
    frames = {}
    for n in (1, 2, 3):
        df = read_tsv(split_dir / f"{split}_source{n}.tsv")
        missing = set(SOURCE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{split}_source{n}.tsv is missing columns {sorted(missing)}")
        frames[n] = df[SOURCE_COLUMNS].reset_index(drop=True)
    pool = pd.concat([frames[2], frames[3]], ignore_index=True)
    return frames[1], pool


def parse_id_list(cell):
    return [x for x in cell.split(",") if x] if cell else []


def read_id_lists(path):
    """Read a two-column ID-list TSV (ground truth or our outputs) into {s1_id: [ids]}."""
    df = read_tsv(path)
    key, value = df.columns[:2]
    return {k: parse_id_list(v) for k, v in zip(df[key], df[value])}


def write_id_lists(path, header, s1_ids, lists):
    """Write one row per Source 1 ID, in the given order, with unquoted comma-joined lists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1 in s1_ids:
            ids = list(dict.fromkeys(lists.get(s1, [])))
            f.write(f"{s1}\t{','.join(ids)}\n")


def write_pair_lists(path, header, s1_ids, i, j, pool_ids):
    """Write one row per Source 1 row index, listing pool_ids[j] for its pairs (i, j).

    Pairs must be unique; within a row, IDs keep the order of the input arrays.
    Streams from arrays, so it never builds a per-entity dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(i, kind="stable")
    ids = np.asarray(pool_ids, dtype=object)[np.asarray(j)[order]]
    bounds = np.searchsorted(np.asarray(i)[order], np.arange(len(s1_ids) + 1))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for r, s1 in enumerate(s1_ids):
            f.write(f"{s1}\t{','.join(ids[bounds[r]:bounds[r + 1]])}\n")
