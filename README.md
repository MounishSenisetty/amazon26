# amazon26: Business Entity Resolution

**Link every Source 1 business record to its matching Source 2 and Source 3 records. The records share no IDs, and names and addresses are noisy across US, India and unseen countries. The pipeline is tuned for the challenge metric, macro F<sub>0.5</sub>.**

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)
![Metric](https://img.shields.io/badge/metric-macro%20F0.5-6f42c1)
![Output](https://img.shields.io/badge/output-TSV-informational)
![Models](https://img.shields.io/badge/pretrained%20models-none-success)
![Data](https://img.shields.io/badge/external%20lookups-none-critical)

This is our solution to the **ML Challenge 2026 Business Entity Resolution Challenge**. The full problem statement is in [`docs/CHALLENGE.md`](docs/CHALLENGE.md).

- **Candidate generation (blocking):** for each Source 1 record, the pipeline takes the top 15 Source 2/3 records by IDF-weighted overlap of hashed name keys (words and word bigrams) and the top 10 by address keys (words, word bigrams and postcode × name-prefix composites), then unions the two lists. Only keys found in at most 3,000 pool records are searched, so blocking scales to millions of records. The result is written to `candidate_pairs.tsv`.
- **Matching model:** a gradient-boosted tree classifier, trained from scratch, scores every candidate on 26 name, address and context features.
- **Metric-aware decision:** the threshold is tuned for macro F<sub>0.5</sub> *with singletons included*, and ties go to the higher threshold. When the training labels show that no Source 2/3 record belongs to two Source 1 entities, each record is also assigned to at most one Source 1 entity. The result is written to `matching_results.tsv`.
- **Open-set countries:** no code path is keyed on `country`. Accent folding and multilingual legal forms (Pvt/Ltd, Corp/Inc, SARL/SAS) let `France`, which appears in test only, go through the same pipeline.
- **Offline and compliant:** no external APIs, geocoders or registries are used, and there are no pretrained models or network calls.

Methodology and design rationale are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Contents

- [Quickstart](#quickstart)
- [Data](#data)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Building the submission package](#building-the-submission-package)
- [Running on Kaggle](#running-on-kaggle)
- [Rules we comply with](#rules-we-comply-with)
- [Project layout](#project-layout)
- [Documentation](#documentation)

---

## Quickstart

### Prerequisites

| Requirement | Why |
| --- | --- |
| Python ≥ 3.11 and `pip` | Required by the pinned `pandas` 3 and `numpy` 2.4 in `requirements.txt`. |
| The challenge `dataset/` folder | Not included in this repo (it is git-ignored). Copy it from the organisers' `student_resource/` bundle. |
| `utils/validate_submission.py` | The organisers' format checker, from `student_resource/utils/`. It is recommended before every upload but not needed to run the pipeline. |
| `zip` | Used only to build the final submission archive. |

### Run end-to-end

```bash
git clone https://github.com/MounishSenisetty/amazon26.git
cd amazon26

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Put the organiser bundle in place
cp -r /path/to/student_resource/dataset ./dataset
cp -r /path/to/student_resource/utils   ./utils

# Train, tune the threshold on a held-out split, then predict on test
python -m src.cli run --data-dir dataset --out-dir output

# Run the organisers' validator before uploading
python3 utils/validate_submission.py \
    --matching  output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir  dataset/test
```

`run` prints a JSON report: the validation scores from training, then a summary of the test predictions with `"checks_passed": true`. The organisers' validator prints `PASS` and exits with code 0. Upload `output/matching_results.tsv` to the Portal.

---

## Data

All files are **tab-separated**, because addresses and ID lists contain commas:

```
dataset/
├── train/
│   ├── train_source1.tsv        # S1: deduplicated reference records
│   ├── train_source2.tsv        # S2 records
│   ├── train_source3.tsv        # S3 records
│   └── train_ground_truth.tsv   # source1_entity_id → matched_entity_ids
└── test/
    ├── test_source1.tsv         # every entity here must appear in the output
    ├── test_source2.tsv
    └── test_source3.tsv
```

Every source file has the same four columns:

| Column | Notes |
| --- | --- |
| `entity_id` | The prefix `S1-`, `S2-` or `S3-` tells you the source. There is no separate source column. |
| `business_name` | May contain abbreviations (Pvt/Private, Corp/Corporation), legal-suffix drift, DBA names, `&`/`and`, reordered words or typos. |
| `business_address` | Components may be missing, in a different order, transliterated or abbreviated (Rd/Road), or given as landmarks ("Near SBI ATM"). |
| `country` | `US` and `India` in train. Test adds `France`. Treat the column as an open set. |

If you explore the data yourself, read it the way the pipeline does ([`src/tsv.py`](src/tsv.py)). Otherwise, empty ID lists become `NaN`, and a name like `NA` becomes a missing value:

```python
from src.tsv import read_tsv, read_id_lists

s1 = read_tsv("dataset/train/train_source1.tsv")               # all columns str, '' kept as ''
gold = read_id_lists("dataset/train/train_ground_truth.tsv")   # {"S1-00001": ["S2-00047", ...], ...}
```

---

## Outputs

`predict` (and `run`) write two TSV files to `--out-dir`:

| File | Columns | Scored? | Contents |
| --- | --- | --- | --- |
| `matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` | **Yes.** This is the leaderboard file. | Final matches. |
| `candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` | No. Organisers use it to audit blocking recall and reduction ratio. | Every pair the classifier scored. |

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

The pipeline guarantees these invariants and checks them after every `predict`. `python -m src.cli check` re-runs the same checks:

- There is exactly one row per `test_source1.tsv` entity, in file order, and no `source1_entity_id` repeats.
- ID lists contain only `S2-`/`S3-` IDs that exist in the test files. There are no duplicates and no quoting.
- A singleton row has an empty second column.
- Every matched ID also appears in the same entity's candidate list, so matches ⊆ candidates.

---

## Configuration

The pipeline is configured entirely with CLI flags (`python -m src.cli <command> --help`). It reads **no API keys or credentials**, because the challenge forbids external lookups.

| Flag | Commands | Default | Description |
| --- | --- | --- | --- |
| `--data-dir` | all | `dataset` | Folder that contains `train/` and `test/`. |
| `--out-dir` | `predict`, `run`, `check` | `output` | Where `matching_results.tsv` and `candidate_pairs.tsv` are written or read. |
| `--model-dir` | `train`, `evaluate`, `predict`, `run` | `artifacts` | Holds `model.joblib`, `model_val.joblib`, `config.json` (threshold, blocking k, validation report) and `split.json`. |
| `--threshold` | `train`, `evaluate`, `predict`, `run` | *tuned* | Overrides the F<sub>0.5</sub>-optimal threshold stored in `config.json`. |
| `--n-jobs` | `train`, `evaluate`, `predict`, `run` | `0` (all CPUs) | Worker processes for normalisation, key hashing, blocking and set features, and RapidFuzz threads. |
| `--max-train-entities` | `train`, `run` | `400000` | Source 1 train entities sampled (seeded) for fitting and validation. Blocking and the context features still cover every record. `0` uses all. |
| `--model` | `train`, `run` | `hgb` | Classifier: `hgb` (scikit-learn HistGradientBoosting, CPU) or `xgboost` (XGBoost, Apache-2.0). |
| `--device` | `train`, `run` | `cpu` | Where XGBoost trains. `cuda` uses the GPU and falls back to the CPU when none is visible; the log names the device actually used. The saved model always predicts on the CPU. |
| `--val-frac` | `train`, `run` | `0.2` | Fraction of the sampled Source 1 train entities (with their gold matches) held out for threshold tuning. `0` skips tuning and uses 0.5. |
| `--seed` | `train`, `run` | `42` | Seed for the split and for the classifier. |
| `--k-name` | `train`, `run` | `15` | Name-key neighbours per Source 1 entity. |
| `--k-addr` | `train`, `run` | `10` | Address-key neighbours per Source 1 entity. |
| `--max-df` | `train`, `run` | `3000` | Blocking only searches keys found in at most this many pool records. Higher values raise recall, time and memory. |
| `--fallback-df` | `train`, `run` | `30000` | A record whose keys are all more frequent than `--max-df` is still blocked on its rarest key, if that key is in at most this many pool records. |
| `--diagnose-blocking` | `train`, `run` | off | Re-blocks the validation entities with 2× `k` and 3× `--max-df`, reports pair completeness for each, counts missed gold pairs that share no name or address key, and writes example misses to `blocking_misses.tsv` in `--model-dir`. |

The blocking settings are saved in `config.json`, so `evaluate` and `predict` reuse them automatically.

| Environment variable | Suggested value | Why |
| --- | --- | --- |
| `OMP_NUM_THREADS` | number of cores to use | Caps the OpenMP threads the gradient-boosting model uses on shared machines. |

For the same data and flags, outputs are deterministic: the split and model are seeded, and candidate and match lists are written in a fixed order.

---

## Usage

```bash
# 1. Fit on train. Hold out 20% of S1 entities, tune the threshold, refit on all, save to artifacts/
python -m src.cli train --data-dir dataset --model-dir artifacts --val-frac 0.2 --seed 42

# 2. (Optional) Re-score the held-out split: macro F0.5/P/R, blocking metrics, per-country breakdown.
#    train already stores this report in artifacts/config.json; evaluate recomputes it from scratch.
python -m src.cli evaluate --data-dir dataset --model-dir artifacts

# 3. Predict on test and write both submission files (runs the rule checks)
python -m src.cli predict --data-dir dataset --model-dir artifacts --out-dir output

# One-shot equivalent of 1 → 3
python -m src.cli run --data-dir dataset --out-dir output

# Re-check existing output files against the submission rules (prints PASS/FAIL)
python -m src.cli check --data-dir dataset --out-dir output

# Try a stricter threshold without retraining
python -m src.cli evaluate --data-dir dataset --model-dir artifacts --threshold 0.7
```

Progress logs go to stderr, and JSON reports go to stdout. For example, `python -m src.cli evaluate … > val.json` captures only the metrics.

---

## Evaluation

The leaderboard metric is **F<sub>β</sub> with β = 0.5, macro-averaged over every Source 1 entity, singletons included** ([`src/metrics.py`](src/metrics.py)):

```
F_0.5 = (1.25 × P × R) / (0.25 × P + R)
```

| Prediction | Truth | Entity score |
| --- | --- | --- |
| empty | empty | 1.0 |
| non-empty | empty | 0.0 |
| empty | non-empty | 0.0 |
| `S2-00047,S2-00193,S3-00812` | `S2-00047,S3-00812` | P = 2/3, R = 1 → **0.714** |

No test labels are released, so `evaluate` scores the held-out training entities and reports:

| Key | Meaning |
| --- | --- |
| `f05`, `precision`, `recall` | Macro scores on the held-out Source 1 entities. |
| `pair_completeness` | Share of true pairs that survive blocking. This is the recall ceiling. |
| `reduction_ratio` | 1 − candidate pairs / (entities × pool size). |
| `mean_candidates_per_entity` | Average candidate list length. |
| `by_country` | Macro scores per `country` label. |

France is absent from train, so no validation number covers it directly.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests generate a small synthetic dataset in the challenge layout ([`tests/synthetic.py`](tests/synthetic.py)). Like the real test set, it has a France slice in test only, so the tests don't need the organiser data.

| Test module | Covers |
| --- | --- |
| `tests/test_metrics.py` | Macro F<sub>0.5</sub>: the 0.714 worked example, the singleton truth table, pair completeness and reduction ratio. |
| `tests/test_tsv.py` | Empty ID lists, row order, de-duplication, and literal `NA`, commas and quotes in fields. |
| `tests/test_normalize.py` | Legal-form canonicalisation, accent and `&` folding, address abbreviations and postcode extraction. |
| `tests/test_pipeline.py` | End-to-end `run`: every test entity including France has a row, matches ⊆ candidates, the rule checks pass, and `check` flags broken files. |

The synthetic data is intentionally easy. Use the tests for correctness, not for model quality. Always finish with `utils/validate_submission.py` on the real outputs.

---

## Building the submission package

The organisers require a single `<team_name>_submission.zip`:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md          # this file
│       └── requirements.txt
└── Documentation_template.md  # docs/METHODOLOGY.md
```

Build it from the repo root. The script uses only the Python standard library and does not need the `zip` command:

```bash
python scripts/make_submission.py --team <team_name>
# -> ./<team_name>_submission.zip   (options: --out-dir output  --dest .  --methodology docs/METHODOLOGY.md)
```

Before you submit, unzip the archive into a clean directory and `cd code/business_entity_resolution`. Then create a fresh virtualenv, `pip install -r requirements.txt`, and run `python -m src.cli run --data-dir /path/to/dataset --out-dir output`. Check that it reproduces both files.

---

## Running on Kaggle

[`kaggle/amazon26_kaggle.ipynb`](kaggle/amazon26_kaggle.ipynb) runs the whole flow in a Kaggle notebook and leaves **`<TEAM_NAME>_submission.zip`** in `/kaggle/working`, where you can download it from the **Output** tab.

1. **Create the notebook.** On kaggle.com choose *Create → New Notebook*, then *File → Import Notebook* and upload `kaggle/amazon26_kaggle.ipynb`.
2. **Add the challenge data.** Choose *Add Input → Upload → New Dataset* and upload the organisers' `student_resource/` folder (or just `dataset/`), keeping it **private**. The notebook finds `train/train_source1.tsv` anywhere under `/kaggle/input`. If `utils/validate_submission.py` is included, the notebook runs it too.
3. **Give it the code.** Either turn on *Settings → Internet* so it clones `REPO_URL` @ `REPO_BRANCH`, or upload this repo as a second dataset. For a private repo, add a `GITHUB_TOKEN` secret under *Add-ons → Secrets*.
4. **Set `TEAM_NAME`** in the first code cell, then choose *Run All*.

The notebook installs the pinned `requirements.txt` into a virtualenv (or a `pip --target` folder when Kaggle's Python has no `venv`), so the outputs match what reviewers reproduce from the zip. Without internet it falls back to Kaggle's preinstalled pandas and scikit-learn, which the pipeline also supports. It then runs `train` and `predict` as separate processes (so train memory is released before test loads), copies the validation report that `train` wrote, validates the outputs and builds the zip. `/kaggle/working` also receives `output/` (the two TSVs), `artifacts/` (models and `config.json`) and `validation_report.json`, which holds the numbers for `docs/METHODOLOGY.md`.

**Accelerator:** a GPU session is recommended. When the notebook sees a GPU (`USE_GPU = True`), it trains with `--model xgboost --device cuda` on a larger sample (`GPU_TRAIN_ENTITIES`, default 1,000,000 Source 1 entities). Normalisation, blocking and the pair features are sparse, string-heavy work, so they run on the session's CPU cores either way. Without a GPU the notebook keeps the CPU defaults.

**Memory:** every stage is chunked, and the log prints peak RAM after each one. If a session still runs out of memory (the cell fails with exit code `-9`), lower `--max-train-entities`, `--k-name`/`--k-addr` or `--max-df` through `EXTRA_ARGS`.

---

## Rules we comply with

- **No external data lookup.** We use no entity-resolution APIs, government registries, geocoding APIs or internet augmentation. The pipeline makes no network calls and learns only from the provided training data.
- **Model licence.** The final model is a gradient-boosted tree ensemble trained from scratch. It uses no pretrained weights and is far below 8B parameters. Library licences (BSD-3-Clause, MIT, Apache-2.0 for XGBoost) are listed in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md#models-and-licences).
- **Output format.** The pipeline follows the rules in [Outputs](#outputs) and checks them on every run.

---

## Project layout

```
.
├── src/
│   ├── cli.py         # entry point: train | evaluate | predict | run | check
│   ├── tsv.py         # TSV reading, ID-list writing
│   ├── normalize.py   # name/address normalisation (country-agnostic)
│   ├── blocking.py    # hashed-key IDF top-k candidate generation
│   ├── features.py    # 26 pair features
│   ├── matcher.py     # classifier, F0.5 threshold tuning, exclusive selection
│   ├── metrics.py     # macro F0.5, pair completeness, reduction ratio
│   ├── checks.py      # submission-rule checks
│   └── parallel.py    # fork-based parallel map over index ranges
├── tests/             # pytest suite + synthetic dataset generator
├── scripts/make_submission.py   # builds <team_name>_submission.zip
├── kaggle/amazon26_kaggle.ipynb  # Kaggle notebook: run everything, output the zip
├── docs/
├── requirements.txt
└── requirements-dev.txt
```

---

## Documentation

| Document | Read it if you want to… |
| --- | --- |
| [`docs/CHALLENGE.md`](docs/CHALLENGE.md) | read the organisers' full problem statement (verbatim). |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | understand the approach, blocking, features, model and threshold (the filled-in submission template). |
| [`docs/DOCUMENTATION_STRATEGY.md`](docs/DOCUMENTATION_STRATEGY.md) | see the docs plan and which guides are still to be written. |
