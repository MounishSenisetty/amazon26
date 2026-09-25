"""Our own checks of the submission rules in docs/CHALLENGE.md.

The organisers' utils/validate_submission.py remains the authority; this
module lets the pipeline fail fast and runs without the organiser bundle.
"""

from pathlib import Path

from .tsv import CANDIDATE_HEADER, MATCH_HEADER, parse_id_list, read_tsv


def _check_file(path, header, s1_ids, pool_ids, label):
    issues = []
    df = read_tsv(path)
    if tuple(df.columns) != header:
        return [f"{label}: header must be {'<TAB>'.join(header)}, got {list(df.columns)}"], {}
    key, value = header
    dup = df[key][df[key].duplicated()].unique()
    if len(dup):
        issues.append(f"{label}: duplicate source1_entity_id rows, e.g. {list(dup[:3])}")
    missing = s1_ids - set(df[key])
    if missing:
        issues.append(f"{label}: {len(missing)} test Source 1 entities missing, e.g. {sorted(missing)[:3]}")
    extra = set(df[key]) - s1_ids
    if extra:
        issues.append(f"{label}: {len(extra)} unknown source1_entity_id values, e.g. {sorted(extra)[:3]}")
    lists = {}
    for s1, cell in zip(df[key], df[value]):
        ids = parse_id_list(cell)
        lists[s1] = set(ids)
        if len(ids) != len(set(ids)):
            issues.append(f"{label}: duplicate IDs in the list for {s1}")
        bad = [x for x in ids if x not in pool_ids]
        if bad:
            issues.append(f"{label}: {s1} lists IDs that are not test Source 2/3 records, e.g. {bad[:3]}")
    return issues, lists


def check_outputs(out_dir, s1, pool):
    """Return a list of rule violations (empty when both files are valid)."""
    out_dir = Path(out_dir)
    s1_ids, pool_ids = set(s1["entity_id"]), set(pool["entity_id"])
    issues = []
    lists = {}
    for name, header in (("matching_results.tsv", MATCH_HEADER), ("candidate_pairs.tsv", CANDIDATE_HEADER)):
        path = out_dir / name
        if not path.exists():
            issues.append(f"{name}: file not found in {out_dir}")
            continue
        found, lists[name] = _check_file(path, header, s1_ids, pool_ids, name)
        issues += found
    if len(lists) == 2:
        for s1_id, matched in lists["matching_results.tsv"].items():
            outside = matched - lists["candidate_pairs.tsv"].get(s1_id, set())
            if outside:
                issues.append(f"{s1_id}: matched IDs missing from candidate_pairs.tsv, e.g. {sorted(outside)[:3]}")
    return issues
