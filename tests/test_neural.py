"""--neural end to end with a tiny local BERT (no download); skipped without torch/transformers."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.checks import check_outputs  # noqa: E402
from src.cli import main  # noqa: E402
from src.tsv import read_id_lists, read_sources  # noqa: E402
from tiny_bert import make_tiny_bert  # noqa: E402


@pytest.fixture(scope="module")
def neural_run(dataset, tmp_path_factory):
    work = tmp_path_factory.mktemp("neural")
    model = make_tiny_bert(work / "tiny_bert")
    main(["run", "--data-dir", str(dataset), "--model-dir", str(work / "artifacts"), "--out-dir", str(work / "output"),
          "--neural", "--neural-model", str(model), "--k-emb", "5", "--model", "xgboost"])
    return work


def test_neural_outputs_satisfy_rules_and_keep_matches_in_candidates(dataset, neural_run):
    s1, pool = read_sources(dataset / "test", "test")
    assert check_outputs(neural_run / "output", s1, pool) == []
    matches = read_id_lists(neural_run / "output" / "matching_results.tsv")
    assert list(matches) == list(s1["entity_id"])


def test_neural_saves_models_and_reports_both_stages(neural_run):
    art = neural_run / "artifacts"
    config = json.loads((art / "config.json").read_text())
    assert config["neural"]["k_emb"] == 5 and 0 < config["neural"]["tau"] <= 0.05
    report = config["report"]
    assert {"validation", "validation_stage1"} <= set(report)
    assert report["ce_entities"] > 0 and report["bienc_entities"] > 0
    for path in ("bi_encoder", "cross_encoder", "model_stage1.joblib", "model_stage2.joblib"):
        assert (art / path).exists()
