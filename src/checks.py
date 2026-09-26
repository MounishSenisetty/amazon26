"""Our own checks of the submission rules in docs/CHALLENGE.md.

The organisers' utils/validate_submission.py remains the authority; this
module lets the pipeline fail fast and runs without the organiser bundle.
"""

from pathlib import Path

from .tsv import CANDIDATE_HEADER, MATCH_HEADER, parse_id_list, read_tsv


def _check_file(path, header, s1_ids, pool_ids, label, on_row):
    issues = []
    df = read_tsv(path)
    if tuple(df.columns) != header:
        return [f"{label}: header must be {'<TAB>'.join(header)}, got {list(df.columns)}"]
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
    for s1, cell in zip(df[key], df[value]):
        ids = parse_id_list(cell)
        id_set = set(ids)
        if len(ids) != len(id_set):
            issues.append(f"{label}: duplicate IDs in the list for {s1}")
        bad = [x for x in ids if x not in pool_ids]
        if bad:
            issues.append(f"{label}: {s1} lists IDs that are not test Source 2/3 records, e.g. {bad[:3]}")
        on_row(s1, id_set)
    return issues


def check_outputs(out_dir, s1, pool):
    """Return a list of rule violations (empty when both files are valid)."""
    out_dir = Path(out_dir)
    s1_ids, pool_ids = set(s1["entity_id"]), set(pool["entity_id"])
    paths = {name: out_dir / name for name in ("matching_results.tsv", "candidate_pairs.tsv")}
    issues = [f"{name}: file not found in {out_dir}" for name, path in paths.items() if not path.exists()]
    if issues:
        return issues
    matched = {}
    issues += _check_file(paths["matching_results.tsv"], MATCH_HEADER, s1_ids, pool_ids,
                          "matching_results.tsv", lambda s1_id, ids: matched.__setitem__(s1_id, ids) if ids else None)

    def subset(s1_id, cands):
        outside = matched.pop(s1_id, set()) - cands
        if outside:
            issues.append(f"{s1_id}: matched IDs missing from candidate_pairs.tsv, e.g. {sorted(outside)[:3]}")

    rows_read = []
    issues += _check_file(paths["candidate_pairs.tsv"], CANDIDATE_HEADER, s1_ids, pool_ids,
                          "candidate_pairs.tsv", lambda s1_id, cands: (rows_read.append(1), subset(s1_id, cands)))
    if rows_read:  # the candidate file was readable, so leftovers have no candidate row at all
        for s1_id, ids in matched.items():
            issues.append(f"{s1_id}: matched IDs missing from candidate_pairs.tsv, e.g. {sorted(ids)[:3]}")
    return issues
