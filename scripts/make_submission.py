"""Build <team_name>_submission.zip in the layout required by docs/CHALLENGE.md.

    python scripts/make_submission.py --team <team_name>

Layout:
    output/matching_results.tsv
    output/candidate_pairs.tsv
    code/business_entity_resolution/{src/, README.md, requirements.txt}
    Documentation_template.md   (from docs/METHODOLOGY.md)

Uses only the standard library, so it runs anywhere the pipeline does (including Kaggle).
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUT_FILES = ("matching_results.tsv", "candidate_pairs.tsv")


def build(team, out_dir, dest_dir, methodology):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", team):
        sys.exit(f"team name {team!r} must only contain letters, digits, '_', '-' or '.'")
    out_dir, dest_dir = Path(out_dir), Path(dest_dir)
    missing = [f for f in OUTPUT_FILES if not (out_dir / f).is_file()]
    if missing:
        sys.exit(f"missing {missing} in {out_dir}: run `python -m src.cli run` first")

    entries = [(out_dir / f, f"output/{f}") for f in OUTPUT_FILES]
    code = "code/business_entity_resolution"
    for path in sorted((REPO / "src").rglob("*.py")):
        entries.append((path, f"{code}/{path.relative_to(REPO).as_posix()}"))
    entries += [
        (REPO / "README.md", f"{code}/README.md"),
        (REPO / "requirements.txt", f"{code}/requirements.txt"),
        (Path(methodology), "Documentation_template.md"),
    ]

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{team}_submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in entries:
            zf.write(src, arcname)
    return zip_path, [a for _, a in entries]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", required=True, help="registered team name; the zip is <team>_submission.zip")
    parser.add_argument("--out-dir", default="output", help="folder with the two pipeline TSVs (default: output)")
    parser.add_argument("--dest", default=".", help="where to write the zip (default: current directory)")
    parser.add_argument("--methodology", default=str(REPO / "docs" / "METHODOLOGY.md"),
                        help="filled-in methodology document (default: docs/METHODOLOGY.md)")
    args = parser.parse_args(argv)
    zip_path, names = build(args.team, args.out_dir, args.dest, args.methodology)
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1024:.0f} KiB)")
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
