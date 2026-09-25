<!--
  DRAFT README — promote to /README.md once src/cli.py implements the commands below.
  Every path, filename, column name and validator flag here comes from the
  ML Challenge 2026 problem statement (docs/CHALLENGE.md). The `python -m src.cli …`
  interface is the proposed pipeline contract; see docs/DOCUMENTATION_STRATEGY.md §4.
  This file is also shipped as code/business_entity_resolution/README.md in the
  submission zip, so every command must work from that folder with --data-dir given.
-->

# amazon26: Business Entity Resolution

**Link every Source 1 business record to its matching Source 2 and Source 3 records. The records share no IDs, and names and addresses are noisy across US, India and unseen countries. The pipeline is tuned for the challenge metric, macro F<sub>0.5</sub>.**

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![Metric](https://img.shields.io/badge/metric-macro%20F0.5-6f42c1)
![Output](https://img.shields.io/badge/output-TSV-informational)
![Models](https://img.shields.io/badge/models-MIT%20%2F%20Apache--2.0%2C%20%E2%89%A48B-success)
![Data](https://img.shields.io/badge/external%20lookups-none-critical)

This is our solution to the **ML Challenge 2026 Business Entity Resolution Challenge**. The full problem statement is in [`docs/CHALLENGE.md`](docs/CHALLENGE.md).

- **Candidate generation (blocking):** narrows about |S1| × (|S2| + |S3|) possible pairs down to a small candidate set for each Source 1 entity. The candidate set is written to `candidate_pairs.tsv`.
- **Matching model:** scores each candidate pair. A decision threshold picked on held-out data for F<sub>0.5</sub> then keeps the likely matches, which are written to `matching_results.tsv`.
- **Singleton-aware:** an empty match list is a valid prediction and scores 1.0 when correct. The threshold is tuned with singletons included.
- **Open-set countries:** `country` is treated as a free-text label. The test set contains `France`, which is not in the training data. No part of the pipeline filters or one-hot encodes `{US, India}`.
- **Offline and compliant:** no external APIs, geocoders or business registries are used. All models are MIT or Apache-2.0 licensed and have at most 8B parameters.

Methodology, blocking design and error analysis are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

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
- [Rules we comply with](#rules-we-comply-with)
- [Documentation](#documentation)

---

## Quickstart

### Prerequisites

| Requirement | Why |
| --- | --- |
| Python ≥ 3.10 and `pip` | Needed to run the pipeline. Versions are pinned in `requirements.txt`. |
| The challenge `dataset/` folder | This repo does not include it. Copy it from the organisers' `student_resource/` bundle. |
| `utils/validate_submission.py` | The organisers' format checker, copied from `student_resource/`. It uses only the standard library. |
| `zip` | Used only to build the final submission archive. |

### Run end-to-end

```bash
git clone https://github.com/MounishSenisetty/amazon26.git
cd amazon26

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Put the organiser data in place (not committed to git)
cp -r /path/to/student_resource/dataset ./dataset
cp -r /path/to/student_resource/utils   ./utils

# Train, choose the threshold on a held-out split, then predict on test
python -m src.cli run --data-dir dataset --out-dir output

# Check both files against the submission rules before uploading
python3 utils/validate_submission.py \
    --matching  output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir  dataset/test
```

A successful run prints `PASS` and exits with code 0. Upload `output/matching_results.tsv` to the Portal.

---

## Data

All files are **tab-separated**, because addresses and ID lists contain commas. Put them in this layout:

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

Read the files with an explicit separator. Keep IDs as strings, and don't let pandas turn empty ID lists into `NaN`:

```python
import pandas as pd

s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)

gold = {r.source1_entity_id: set(filter(None, r.matched_entity_ids.split(",")))
        for r in gt.itertuples()}
```

Data profiling and the noise catalogue are in [`docs/DATA.md`](docs/DATA.md).

---

## Outputs

The pipeline writes two TSV files to `--out-dir` (default `output/`):

| File | Columns | Scored? | Contents |
| --- | --- | --- | --- |
| `matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` | **Yes.** This is the leaderboard file. | Final matches. |
| `candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` | No. Organisers use it to audit blocking recall and reduction ratio. | The exact set the matching model scored. |

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

The pipeline guarantees these invariants, which are checked by tests and by the organisers' validator:

- Each `test_source1.tsv` entity gets exactly one row, and no `source1_entity_id` repeats.
- ID lists contain only `S2-`/`S3-` IDs that exist in the test files. There are no duplicates and no quoting.
- A singleton row has an empty second column.
- Every matched ID also appears in the same entity's candidate list, so matches ⊆ candidates.

---

## Configuration

The pipeline is configured entirely with CLI flags. It reads **no API keys or credentials**, because the challenge forbids external lookups.

| Flag | Default | Description |
| --- | --- | --- |
| `--data-dir` | `dataset` | Folder that contains `train/` and `test/`. |
| `--out-dir` | `output` | Where `matching_results.tsv` and `candidate_pairs.tsv` are written. |
| `--model-dir` | `artifacts` | Where the fitted model, the chosen threshold and the validation split manifest are saved. |
| `--val-frac` | `0.2` | Fraction of Source 1 train entities (with their gold matches) held out for threshold tuning and evaluation. |
| `--seed` | `42` | Seed for the train/validation split and for model training. |
| `--threshold` | *tuned* | Overrides the F<sub>0.5</sub>-optimal threshold chosen on validation. |

| Environment variable | Recommended value | Why |
| --- | --- | --- |
| `PYTHONHASHSEED` | `0` | Makes set and dict iteration deterministic, so the output files are byte-for-byte reproducible. |
| `HF_HUB_OFFLINE` | `1` (after the first model download) | Stops the run from contacting the network. Only matters if a Hugging Face model is used. |
| `OMP_NUM_THREADS` | number of cores | Caps BLAS threads on shared machines. |

---

## Usage

```bash
# 1. Fit on train. Hold out 20% of S1 entities; save model, threshold and split manifest.
python -m src.cli train --data-dir dataset --model-dir artifacts --val-frac 0.2 --seed 42

# 2. Score the held-out split: blocking recall, reduction ratio, macro F0.5 / P / R
python -m src.cli evaluate --data-dir dataset --model-dir artifacts

# 3. Predict on test and write both submission files
python -m src.cli predict --data-dir dataset --model-dir artifacts --out-dir output

# One-shot equivalent of 1 → 3
python -m src.cli run --data-dir dataset --out-dir output
```

Quick sanity checks on the outputs:

```bash
# Row count must equal the number of test S1 entities (both files include a header)
wc -l dataset/test/test_source1.tsv output/matching_results.tsv output/candidate_pairs.tsv

# How many S1 entities are predicted as singletons?
awk -F'\t' 'NR>1 && $2==""' output/matching_results.tsv | wc -l
```

---

## Evaluation

The leaderboard metric is **F<sub>β</sub> with β = 0.5, macro-averaged over every Source 1 entity, singletons included**:

```
F_0.5 = (1.25 × P × R) / (0.25 × P + R)
```

| Prediction | Truth | Entity score |
| --- | --- | --- |
| empty | empty | 1.0 |
| non-empty | empty | 0.0 |
| empty | non-empty | 0.0 |
| `S2-00047,S2-00193,S3-00812` | `S2-00047,S3-00812` | P = 2/3, R = 1 → **0.714** |

No test labels are released. `src.cli evaluate` therefore scores a held-out split of the training data, and also reports blocking **pair completeness** (the recall ceiling) and **reduction ratio**. France appears only in test, so [`docs/EVALUATION.md`](docs/EVALUATION.md) also describes a leave-one-country-out check (train US → validate India, and the reverse) as a proxy for unseen-country generalisation.

---

## Testing

```bash
pip install pytest
pytest -q
```

| Test module | Covers |
| --- | --- |
| `tests/test_io.py` | TSV round-trip, empty ID lists, commas inside addresses, no quoting in output. |
| `tests/test_metrics.py` | Macro F<sub>0.5</sub>, including the 0.714 worked example and all singleton cases above. |
| `tests/test_invariants.py` | One row per S1, S2/S3-only IDs, no duplicates, matches ⊆ candidates. |
| `tests/test_open_country.py` | A `France` record passes through every stage without errors and appears in the output. |

Always finish with `utils/validate_submission.py` (see [Quickstart](#run-end-to-end)). Our tests do not replace it.

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
└── Documentation_template.md  # filled in from docs/METHODOLOGY.md
```

```bash
TEAM=<team_name>
STAGE=build/submission
rm -rf "$STAGE" && mkdir -p "$STAGE/output" "$STAGE/code/business_entity_resolution"
cp output/matching_results.tsv output/candidate_pairs.tsv "$STAGE/output/"
cp -r src README.md requirements.txt "$STAGE/code/business_entity_resolution/"
cp docs/METHODOLOGY.md "$STAGE/Documentation_template.md"
(cd "$STAGE" && zip -r "../../${TEAM}_submission.zip" .)
```

Before you submit, unzip the archive into a clean directory. Then create a fresh virtualenv, `pip install -r requirements.txt`, and run `python -m src.cli run --data-dir /path/to/dataset --out-dir output`. Check that it reproduces both files.

---

## Rules we comply with

- **No external data lookup.** We use no entity-resolution APIs, government registries, geocoding APIs or internet augmentation. The pipeline makes no network calls at inference time, and the only network use is downloading model weights.
- **Model licence.** The final model is MIT or Apache-2.0 licensed with ≤ 8B parameters. Each model we use is listed with its licence and parameter count in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md#appendix).
- **Output format.** The pipeline follows the rules in [Outputs](#outputs). Files that fail validation are not scored.

---

## Documentation

| Document | Read it if you want to… |
| --- | --- |
| [`docs/CHALLENGE.md`](docs/CHALLENGE.md) | read the organisers' full problem statement (verbatim). |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | understand the approach, blocking, features, model, threshold and error analysis (the filled-in submission template). |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | find your way around `src/`, the stage interfaces and the data flow. |
| [`docs/DATA.md`](docs/DATA.md) | see dataset statistics, noise patterns and per-country address formats. |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | reproduce validation numbers, the split protocol and the blocking metrics. |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | see the leaderboard and validation history. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | set up dev tools, follow branch conventions and add a feature or blocking key. |
