# ML Challenge 2026: Business Entity Resolution Solution Template

<!--
  This is the filled-in Documentation_template.md. The packaging step copies it
  to Documentation_template.md at the root of the submission zip. Keep the
  section order: it is the organisers' template.
  Items marked ⏳ need numbers from a run on the real dataset:
  `python -m src.cli train` then `python -m src.cli evaluate` (see README §Usage).
-->

**Team Name:** ⏳ *(team to fill)*  
**Team Members:** ⏳ *(team to fill)*  
**Submission Date:** ⏳

---

## 1. Executive Summary

We use a two-stage pipeline. **Blocking** unions each Source 1 record's nearest Source 2/3 neighbours by character n-gram TF-IDF on normalised names and on normalised addresses. A **gradient-boosted tree classifier**, trained from scratch on 26 string, address and context features, then scores every candidate pair. Final matches are candidates whose probability clears a threshold chosen to maximise **macro F<sub>0.5</sub> with singletons included**. When the training ground truth shows that no Source 2/3 record belongs to two Source 1 entities, a one-owner-per-record rule is also applied. No part of the pipeline is keyed on the country label, so unseen countries such as France go through the same path.

---

## 2. Methodology

### 2.1 Problem Analysis

These points come from the challenge statement. The ⏳ items need EDA on the real data.

- **Name noise:** legal-form drift (Private/Pvt, Limited/Ltd, Corporation/Corp), `&` vs. `and`, punctuation, reordered words, typos, and DBA/trade names that can differ entirely from the legal name.
- **Address noise:** abbreviations (Road/Rd, Street/St), transliteration, missing PIN/ZIP or state, landmark references ("Near SBI ATM"), and reordered components.
- **Open-set country:** train covers US and India, while test adds France. Any country-specific rule learned only from train cannot help France, and a rule that filters on `{US, India}` would break the output.
- **Metric:** macro F<sub>0.5</sub> scores each Source 1 entity equally. A singleton scores 1.0 for an empty prediction and 0.0 for any match. False merges are therefore costly on singletons, and precision is weighted 2× over recall.
- ⏳ Dataset statistics: records per source and country, the singleton share, the distribution of matches per entity, the S2 vs. S3 match share, and the rate of missing postcodes.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier  
**Core Innovation:** The threshold and ownership rule are chosen directly on the challenge metric. The threshold is tuned for macro F<sub>0.5</sub> with singletons included, and ties go to the higher threshold. Exclusive assignment is switched on only when the ground truth shows no pool record shared between Source 1 entities. Every feature is country-agnostic, including accent folding and multilingual legal forms, so the model transfers to unseen countries.

Stages (`src/`):

| Stage | Module | Output |
| --- | --- | --- |
| Read TSVs (`dtype=str`, `keep_default_na=False`, `QUOTE_NONE`) | `tsv.py` | S1 table and the S2+S3 pool |
| Normalise names and addresses | `normalize.py` | `name_norm`, `name_core`, `addr_norm`, postcodes, numbers, `country_norm` |
| Candidate generation | `blocking.py` | Candidate pairs, written to `candidate_pairs.tsv` |
| Pair features | `features.py` | 26 features per pair |
| Classifier, threshold, selection | `matcher.py` | `matching_results.tsv` |
| Metrics | `metrics.py` | Macro F<sub>0.5</sub>/P/R, pair completeness, reduction ratio |
| Rule checks | `checks.py` | List of submission-rule violations |

---

## 3. Candidate Generation (Blocking)

- **Normalisation before blocking:**
  - Unicode NFKD accent stripping and lowercasing; `&` becomes `and`; non-alphanumerics become spaces.
  - Legal forms are mapped to one token each, e.g. `private→pvt`, `limited→ltd`, `corporation→corp`. French and German forms such as SARL, SAS, SA and GmbH are kept as-is.
  - Address abbreviations are canonicalised, e.g. `road→rd`, `street→st`, `boulevard/bd→blvd`, `avenue/av→ave`.
- **Blocking keys used:**
  1. The top **k = 15** pool records by cosine of character 2–4-gram (`char_wb`) TF-IDF on the normalised name.
  2. The top **k = 10** by the same representation on the normalised address.

  The candidate set is the union of the two lists. The TF-IDF vocabulary is fitted on the text of the split being processed; no labels are used. Blocking is deliberately *not* partitioned by country or postcode, so a noisy or missing postcode cannot silently drop a true match.
- **Candidate pairs generated:** ⏳ (from the `predict` report: `candidate_pairs`). At most 25 per Source 1 entity.
- **How we ensured true matches were not lost:**
  - Two independent channels: a DBA name mismatch can still be recovered through the address, and a landmark-only address through the name.
  - Character n-grams tolerate typos and transliteration.
  - `evaluate` reports **pair completeness** (the recall ceiling) and the **reduction ratio** on the held-out split.
  - `--k-name` and `--k-addr` can be raised if pair completeness is too low. Validation pair completeness is ⏳.

---

## 4. Matching Model

**Features used** (`src/features.py`):

- **Name features:**
  - TF-IDF cosine.
  - RapidFuzz `ratio`, `token_set_ratio` and `partial_ratio` on the normalised name.
  - `ratio`, `token_set_ratio`, Jaro-Winkler and token Jaccard on the *core* name (legal forms and stopwords removed).
  - First-token match, acronym match (e.g. "ABC" vs. "Alpha Beta Corp"), legal-form agreement (1, 0, or missing if either side has none), and relative length difference.
- **Address features:** TF-IDF cosine, `token_set_ratio`, `partial_ratio`, token Jaccard, postcode agreement (5/6-digit codes; missing if either side has none), and Jaccard of all numbers in the address.
- **Other:**
  - Same country label, and whether the record comes from Source 3.
  - Context within the Source 1 entity's candidate list: rank and gap to the best name and address similarity, and the number of candidates.
  - Reverse rank: how this Source 1 entity ranks among all Source 1 entities that proposed the same pool record.

**Model type:** scikit-learn `HistGradientBoostingClassifier` (300 iterations, learning rate 0.08, 31 leaves, L2 = 1.0, seed 42), trained from scratch. No pretrained model or weights are used.

**Threshold selection method:**
1. Hold out 20% of Source 1 train entities, together with their gold matches (`--val-frac`, `--seed`). The full Source 2/3 pool stays searchable, exactly as at test time.
2. Fit on the remaining entities' candidate pairs.
3. Sweep thresholds from 0.05 to 0.95 in 0.01 steps, and keep the one with the best **macro F<sub>0.5</sub> over all held-out entities, singletons included**. Ties go to the higher threshold.
4. If the gold has no pool record shared between Source 1 entities, each pool record is kept only for its highest-scoring Source 1 entity.
5. Refit the final model on all training entities with the chosen threshold.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** ⏳ (validation, from `python -m src.cli evaluate`: `f05`, with the per-country breakdown under `by_country`)
- **Blocking:** pair completeness ⏳, reduction ratio ⏳
- **Public leaderboard:** ⏳
- **Common false positives (wrong merges):** ⏳ Expected risk: chains or franchises that share a name and differ only by address.
- **Common false negatives (missed matches):** ⏳ Expected risk: DBA names paired with landmark-only addresses, where both similarity channels are weak.

---

## 6. Conclusion

⏳ Summarise after the real-data results are in.

---

## Appendix

### A. Code Artefacts

The code ships in `code/business_entity_resolution/`:

```
src/
├── cli.py         # entry point: train | evaluate | predict | run | check
├── tsv.py         # TSV reading, ID-list writing
├── normalize.py   # name/address normalisation
├── blocking.py    # TF-IDF top-k candidate generation
├── features.py    # pair features
├── matcher.py     # classifier, threshold tuning, exclusive selection
├── metrics.py     # macro F0.5, pair completeness, reduction ratio
└── checks.py      # submission-rule checks
README.md
requirements.txt
```

To reproduce both `output/matching_results.tsv` and `output/candidate_pairs.tsv`, follow README §Quickstart:

```bash
pip install -r requirements.txt
python -m src.cli run --data-dir /path/to/dataset --out-dir output
```

### Models and licences

| Component | Licence | Parameters | Pretrained? |
| --- | --- | --- | --- |
| HistGradientBoostingClassifier (our trained model) | Trained by us; library scikit-learn BSD-3-Clause | Tree ensemble, ≤ 300 trees × 31 leaves | No |
| TF-IDF vectoriser | scikit-learn BSD-3-Clause | Fitted per run on the split's own text | No |
| RapidFuzz string metrics | MIT | None (deterministic functions) | No |

No external data, APIs, registries or geocoders are used. The pipeline makes no network calls.

### B. Additional Results

⏳ Threshold-vs-F<sub>0.5</sub> curve, per-country scores, and blocking recall as k varies.
