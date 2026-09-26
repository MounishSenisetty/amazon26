"""End-to-end run on the synthetic dataset, which has France in test only."""

import json

import pytest

from src.checks import check_outputs
from src.cli import main
from src.tsv import read_id_lists, read_sources


@pytest.fixture(scope="module")
def run_outputs(dataset, tmp_path_factory):
    work = tmp_path_factory.mktemp("run")
    main(["run", "--data-dir", str(dataset), "--model-dir", str(work / "artifacts"),
          "--out-dir", str(work / "output")])
    return work


def test_outputs_satisfy_submission_rules(dataset, run_outputs):
    s1, pool = read_sources(dataset / "test", "test")
    assert check_outputs(run_outputs / "output", s1, pool) == []


def test_every_test_entity_including_france_has_a_row(dataset, run_outputs):
    s1, _ = read_sources(dataset / "test", "test")
    matches = read_id_lists(run_outputs / "output" / "matching_results.tsv")
    assert "France" in set(s1["country"])
    assert list(matches) == list(s1["entity_id"])


def test_matches_are_subset_of_candidates(run_outputs):
    matches = read_id_lists(run_outputs / "output" / "matching_results.tsv")
    cands = read_id_lists(run_outputs / "output" / "candidate_pairs.tsv")
    assert all(set(m) <= set(cands[k]) for k, m in matches.items())


def test_config_records_tuned_threshold(run_outputs):
    config = json.loads((run_outputs / "artifacts" / "config.json").read_text())
    assert 0.05 <= config["threshold"] <= 0.95
    assert config["report"]["validation"]["pair_completeness"] > 0.9


def test_check_command_flags_violations(dataset, run_outputs, tmp_path, capsys):
    bad = tmp_path / "output"
    bad.mkdir()
    lines = (run_outputs / "output" / "matching_results.tsv").read_text().splitlines()
    (bad / "matching_results.tsv").write_text("\n".join(lines[:-1] + [lines[1]]) + "\n")
    (bad / "candidate_pairs.tsv").write_text((run_outputs / "output" / "candidate_pairs.tsv").read_text())
    with pytest.raises(SystemExit) as exc:
        main(["check", "--data-dir", str(dataset), "--out-dir", str(bad)])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "duplicate source1_entity_id" in out and "missing" in out and out.strip().endswith("FAIL")


def test_train_report_and_evaluate_agree(dataset, run_outputs, capsys):
    report = json.loads((run_outputs / "artifacts" / "config.json").read_text())["report"]["validation"]
    assert set(report["by_country"]) == {"US", "India"}
    capsys.readouterr()
    main(["evaluate", "--data-dir", str(dataset), "--model-dir", str(run_outputs / "artifacts")])
    evaluated = json.loads(capsys.readouterr().out)
    assert evaluated["f05"] == pytest.approx(report["f05"])
    assert evaluated["pair_completeness"] == pytest.approx(report["pair_completeness"])
