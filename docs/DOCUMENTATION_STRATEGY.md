# Documentation Strategy: amazon26

*Audit date: 2026-09-25 · Repo state audited: `main` @ `a6d53e3` ("Initial commit")*

This document explains how the project should be documented, why, and in what order the files should be written. The README blueprint in §4 has since been implemented as the root [`README.md`](../README.md). §5 tracks which action-plan items are done.

---

## 1. Repository audit

### 1.1 What the repository actually contains

| Path | Contents |
| --- | --- |
| `README.md` | One line: `# amazon26` |

That is the entire tracked tree: one commit and one file. There is **no** source code, entry point, configuration, dependency manifest, test, dataset, validator script or licence. No git history shows earlier layouts either.

There were two other inputs to this audit. Neither is in the repo:

| Input | What it is | What it is **not** |
| --- | --- | --- |
| Attached `README.md` | The organisers' **problem statement** for the ML Challenge 2026 Business Entity Resolution Challenge. It defines the data schema, output contract, metric, packaging, and licence and fair-play rules. | A description of *this* project's solution, setup or code. |
| Attached `Documentation_template.md` | The **methodology write-up template** that must be filled in and shipped inside the submission zip. | A README. It is a graded deliverable with a fixed section order. |

### 1.2 What the docs claim vs. what exists

The only real claims come from the challenge statement. The table checks each one against the repo:

| Claim or instruction in the existing docs | Reality in the repo | Gap |
| --- | --- | --- |
| Data lives at `dataset/train/*.tsv` and `dataset/test/*.tsv`. | No `dataset/` directory. | Setup must say where the data comes from (organisers' `student_resource/`) and that it is **not committed**. `.gitignore` must exclude it. |
| "Run `utils/validate_submission.py` from this `student_resource/` directory." | No `utils/`, no `student_resource/`. | The command only works after the organiser bundle is copied in. The README must include that step or the command fails. |
| Zip must contain `code/business_entity_resolution/{src/, README.md, requirements.txt}`. | No `src/`, no `requirements.txt`. | The repo root should **mirror** that folder so packaging is a copy, not a rewrite. The root README is the zip README, so its commands must work from inside that folder with `--data-dir` passed explicitly. |
| `requirements.txt` must pin versions. | Missing. | Blocking for reproducibility review. |
| Methodology doc must describe blocking, model, features, threshold and results. | Template unfilled, and not in the repo. | Add it as `docs/METHODOLOGY.md`. The packaging step copies it back to `Documentation_template.md`. |
| Template Appendix A asks for "the entry point(s) to reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv`". | No entry point. | The README and the methodology appendix must name **the same** command. Define it once (proposed: `python -m src.cli run`). |
| "Hold out a validation split … score it yourself with F<sub>0.5</sub>." | No scorer is provided by organisers or in the repo. | An evaluation module plus `docs/EVALUATION.md` is needed. Without it no reported number is reproducible. |
| Organisers analyse "recall ceiling, reduction ratio" of `candidate_pairs.tsv`. | Nothing computes them. | `evaluate` should report pair completeness and reduction ratio. The template §3 asks for "Candidate pairs generated: [total]". |
| Test includes `France`, which is absent from train. Don't hard-code `{US, India}`. | No code yet. | Record this as a **design invariant** in the README and ARCHITECTURE, and back it with a test (`test_open_country.py`). |
| Final model must be MIT/Apache-2.0 and ≤ 8B params. No external lookups. | No model or licence listing. | Add a compliance section to the README and a model and licence table in the METHODOLOGY appendix. The reviewers audit exactly this. |
| Output ID lists use "no quoting", and empty cells mean singletons. | No I/O code. | Document the pandas pitfall: an empty `matched_entity_ids` becomes `NaN` unless you pass `keep_default_na=False`. Provide a safe read snippet. |

### 1.3 Undocumented or under-specified points to resolve

1. **Project licence.** There is no `LICENSE`. The challenge constrains *model* licences but says nothing about the repo's own. Decide before making the repo public.
2. **Python version.** Nothing pins it. The draft proposes ≥ 3.10. Confirm this when `requirements.txt` is written.
3. **Determinism.** Set and dict ordering of string IDs depends on `PYTHONHASHSEED`. Without fixing it, output files are not byte-identical across runs, which makes the organiser reproduction harder.
4. **Validation protocol.** Train S1 entities can be split, but S2/S3 records form one shared pool. The protocol has to state how negatives are handled so validation mirrors test, where the full pool is searched.
5. **Unseen-country generalisation.** France cannot be validated directly. Leave-one-country-out (US ↔ India) is the only in-data proxy, so document it.

---

## 2. Audience profile and priorities

This repository is a **competition submission**, not a library or a deployed service. "Deployment" here means *producing a valid submission zip*, not running a server. The audiences, in priority order:

| # | Audience | What they need first | Where they find it |
| --- | --- | --- | --- |
| 1 | **Challenge reviewers** reproducing a top-team package. They decide final rankings. | Exact prerequisites → one command → both TSVs. Evidence of licence and fair-play compliance. | README: Quickstart, Outputs, Rules. METHODOLOGY appendix. |
| 2 | **Team members** iterating on blocking and model. | Train/eval/predict commands, the validation protocol, the metric definition, and where each stage lives in `src/`. | README: Usage, Evaluation. `docs/ARCHITECTURE.md`, `docs/EVALUATION.md`, `docs/EXPERIMENTS.md`. |
| 3 | **Future readers** (portfolio, hiring, other ER practitioners). | What problem, what approach, how good. | README headline plus METHODOLOGY summary. |

This order sets the README section order: the reproduction path comes before the explanation, and the long-form method is one link away.

---

## 3. Information architecture

### 3.1 Target tree

```
amazon26/                          ≙ code/business_entity_resolution/ in the zip
├── README.md                      # lean: what, quickstart, contract, config, usage, test, package
├── CONTRIBUTING.md                # dev setup, conventions, how to add a feature/blocking key
├── LICENSE                        # repo licence (decision pending)
├── requirements.txt               # pinned runtime deps (zip requirement)
├── requirements-dev.txt           # pytest, linters
├── .gitignore                     # dataset/, output/, artifacts/, build/, *.zip, .venv/
├── src/                           # all pipeline code (zip requirement)
│   └── cli.py                     # train | evaluate | predict | run
├── tests/
├── utils/validate_submission.py   # copied from organiser bundle, not modified
├── dataset/                       # organiser data (git-ignored)
└── docs/
    ├── CHALLENGE.md               # organiser problem statement, verbatim, source-linked
    ├── METHODOLOGY.md             # filled Documentation_template.md (graded deliverable)
    ├── ARCHITECTURE.md            # module map, stage I/O, data flow diagram
    ├── DATA.md                    # EDA: counts, match-cardinality, noise catalogue per country
    ├── EVALUATION.md              # split protocol, metric code, blocking metrics, LOCO check
    ├── EXPERIMENTS.md             # dated log: change → val F0.5 → public LB
    └── DOCUMENTATION_STRATEGY.md  # this file
```

### 3.2 What goes where (single-source-of-truth rules)

| Topic | Canonical home | Everyone else… |
| --- | --- | --- |
| Challenge rules, schema, metric definition | `docs/CHALLENGE.md` (verbatim) | The README restates only the parts needed to run and validate, and links to the rest. |
| How to run (commands, flags, env vars) | `README.md` | METHODOLOGY Appendix A **links** to the README Usage section instead of repeating commands. |
| Why the approach works, and the numbers | `docs/METHODOLOGY.md` | The README states no scores, so numbers never go stale in two places. |
| Code structure and stage interfaces | `docs/ARCHITECTURE.md` | CONTRIBUTING links here. |
| Score history | `docs/EXPERIMENTS.md` | METHODOLOGY §5 quotes the final row only. |
| Validation protocol and metric implementation | `docs/EVALUATION.md` | README Evaluation gives a 5-line summary and a link. |

**Why the README stays lean:** it is copied into the zip as the reproduction guide. Reviewers read it with a stopwatch, so anything that isn't about running the pipeline belongs in `docs/`.

---

## 4. README blueprint (rationale for the root `README.md`)

| # | Section | Contents | Why here |
| --- | --- | --- | --- |
| 0 | Title plus one-sentence value prop | "Link every Source 1 record to its matching S2/S3 records … optimised for macro F<sub>0.5</sub>." | States the task, the input shape and the success metric in one line. |
| 1 | Badges | Python version, metric, output format, model-licence constraint, "no external lookups". | These are compliance signals the reviewer checks first. No CI or coverage badges until CI exists. |
| 2 | Core features (5 bullets) | Blocking, matching, singleton-aware threshold, open-set country, offline and compliant. | Each bullet maps to a judged requirement or a metric subtlety. |
| 3 | Quickstart | Prerequisites table (incl. organiser bundle), venv, copy `dataset/` and `utils/`, `src.cli run`, organiser validator. | The shortest path from clone to `PASS`. Audience #1. |
| 4 | Data | Tree, 4-column schema with noise notes, safe `read_csv` snippet (`dtype=str, keep_default_na=False`). | Prevents the two most likely silent bugs: a missing `sep="\t"` and `NaN` empty lists. |
| 5 | Outputs | Both files, scored vs. audited, example rows, invariants list. | This is the contract that decides whether a submission is scored at all. |
| 6 | Configuration | CLI flag table and env-var table. States that there are no secrets or API keys. | The pipeline has no config file. Saying "no keys" is itself a compliance statement. |
| 7 | Usage | `train`, `evaluate`, `predict`, `run`, plus awk/wc sanity checks. | Audience #2's inner loop. |
| 8 | Evaluation | Formula, singleton truth table, blocking metrics, LOCO note. | Makes F<sub>0.5</sub> edge cases unambiguous. |
| 9 | Testing | `pytest`, table of test modules to invariants. Validator is still required. | Ties each test to a rule it protects. |
| 10 | Building the submission package | Exact zip tree and a copy-paste packaging script, plus a clean-room reproduction check. | This is the "deployment" for this project. |
| 11 | Rules we comply with | External-lookup ban, model licence/size, format. | Reviewers audit these explicitly. |
| 12 | Documentation index | Table of `docs/*` with a "read it if you want to…" column. | Keeps the README lean without hiding depth. |

**The CLI is now implemented.** `python -m src.cli {train,evaluate,predict,run,check}` exists in `src/cli.py`, and the README flag table matches `--help`. Keep them in sync (see §6).

---

## 5. Action plan

Ordered by dependency. Priority: **P0** blocks a valid, reproducible submission; **P1** is needed for review quality; **P2** helps team velocity.

| # | File / directory | Priority | Contents | Source of truth | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | `.gitignore` | P0 | `dataset/`, `output/`, `artifacts/`, `build/`, `*_submission.zip`, `.venv/`, `__pycache__/` | Organiser data must not be committed. | ✅ Done |
| 2 | `docs/CHALLENGE.md` | P0 | The attached problem statement, verbatim, with a "source: organiser `student_resource/README.md`, retrieved <date>" header. | Attached README. | ✅ Done |
| 3 | `utils/validate_submission.py` | P0 | Copied unmodified from `student_resource/utils/`. | Organiser bundle. | ⛔ Blocked: the organiser file is not in this environment. Copy it in from `student_resource/utils/`. Until then, `python -m src.cli check` enforces the same rules. |
| 4 | `requirements.txt` | P0 | Pinned (`==`) runtime deps, generated from the working venv. | Actual imports in `src/`. | ✅ Done (Python ≥ 3.11 because of pandas 3 / numpy 2.4) |
| 5 | `src/cli.py` (+ stage modules) | P0 | Implements the `train`/`evaluate`/`predict`/`run` contract in README §Usage/§Configuration. | This strategy §4. | ✅ Done (adds a `check` command) |
| 6 | `README.md` | P0 | Promote `docs/README.proposed.md` and delete the draft header comment. | §4. | ✅ Done |
| 7 | `docs/METHODOLOGY.md` | P0 | The filled `Documentation_template.md`, all sections. Appendix A links to README §Usage. Appendix adds a **model / licence / params table**. | Attached template. | 🟡 Methods and licence table filled. Team details and all results (§5, §6, App. B) are marked ⏳ until the pipeline is run on the real dataset. |
| 8 | `docs/EVALUATION.md` | P1 | Split protocol (hold out S1 entities with their gold, search the full S2/S3 pool), macro F<sub>0.5</sub> code with the singleton truth table, pair completeness, reduction ratio, candidates-per-entity distribution, leave-one-country-out procedure. | `src/` evaluation module. | Open |
| 9 | `tests/` | P1 | `test_io.py`, `test_metrics.py` (asserts 0.714 example), `test_invariants.py`, `test_open_country.py`. | README §Testing table. | 🟡 Done early to verify item 5: `test_metrics`, `test_tsv`, `test_normalize`, `test_pipeline` (on a synthetic dataset) |
| 10 | `docs/ARCHITECTURE.md` | P1 | Module map of `src/`, per-stage input/output (DataFrame columns or TSV), a data-flow diagram (normalise → block → featurise → score → threshold → write), extension points. | `src/`. | Open |
| 11 | `docs/DATA.md` | P1 | Row counts per source and country, match cardinality (0/1/many), singleton rate, S2 vs. S3 match share, noise examples per country, address format notes (PIN vs. ZIP). | EDA notebook/script. | Open |
| 12 | `scripts/make_submission.sh` | P1 | The README packaging snippet as a script taking `TEAM` as its argument, followed by an automatic clean-room reproduction in a temp venv. | README §Building. | ✅ Done as `scripts/make_submission.py` (stdlib only), plus `kaggle/amazon26_kaggle.ipynb`, which runs the pipeline and builds the zip on Kaggle |
| 13 | `CONTRIBUTING.md` | P2 | Dev setup (`requirements-dev.txt`), branch naming, "every PR updates EXPERIMENTS.md if val score changes", how to add a blocking key or feature. | Team conventions. | Open (`requirements-dev.txt` already exists) |
| 14 | `docs/EXPERIMENTS.md` | P2 | Table with columns date, commit, change, val P/R/F<sub>0.5</sub>, pair completeness, public LB. | Team runs. | Open |
| 15 | `LICENSE` | P2 | Pick one (e.g. MIT) once the team agrees. | Team decision. | Open (team decision) |

### Suggested sequencing

1. **Now:** items 1–4 (scaffolding, no code decisions needed).
2. **With the first working pipeline:** items 5, 6, 9, then run the validator and get `PASS`.
3. **Before the first leaderboard upload:** item 8, so the uploaded score has a reproducible validation counterpart.
4. **Before final submission:** items 7, 10, 11 and 12, then run the clean-room reproduction.

---

## 6. Keeping docs from rotting

- **Test the README's commands.** A CI job (or `scripts/make_submission.py`) runs the exact Quickstart commands on a tiny fixture dataset under `tests/fixtures/dataset/`, then runs the organiser validator. If the README and the CLI disagree, the build fails.
- **CLI help is authoritative for flags.** When a flag changes, update `--help` and the README Configuration table in the same PR. A test can assert that every flag in the table appears in `python -m src.cli --help`.
- **Numbers live only in METHODOLOGY and EXPERIMENTS**, never in the README.
- **CHALLENGE.md is verbatim.** Never edit it. If organisers publish an update, replace the file and note the date in its header.
