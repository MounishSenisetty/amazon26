import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from make_submission import build  # noqa: E402


def test_zip_has_the_required_layout(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "matching_results.tsv").write_text("source1_entity_id\tmatched_entity_ids\nS1-00001\t\n")
    (out / "candidate_pairs.tsv").write_text("source1_entity_id\tcandidate_entity_ids\nS1-00001\t\n")
    zip_path, _ = build("my_team", out, tmp_path, Path(__file__).resolve().parent.parent / "docs" / "METHODOLOGY.md")

    assert zip_path.name == "my_team_submission.zip"
    names = set(zipfile.ZipFile(zip_path).namelist())
    code = "code/business_entity_resolution"
    assert {"output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
            f"{code}/README.md", f"{code}/requirements.txt", f"{code}/requirements-neural.txt",
            f"{code}/src/cli.py", f"{code}/src/neural.py"} <= names
    assert all(n.startswith(("output/", f"{code}/", "Documentation_template.md")) for n in names)
    assert not any("__pycache__" in n for n in names)
