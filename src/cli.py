"""Command-line entry point: python -m src.cli {train,evaluate,predict,run,check}."""

import argparse
import gc
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import matcher
from .blocking import build_keys, candidate_lists, generate_candidates
from .blocking import diagnose as blocking_diagnose
from .checks import check_outputs
from .features import context_features, pair_features, similarity_features
from .metrics import blocking_scores, macro_scores
from .normalize import prepare
from .parallel import resolve_jobs
from .tsv import CANDIDATE_HEADER, MATCH_HEADER, read_id_lists, read_sources, write_pair_lists


def log(msg):
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    print(f"[{time.strftime('%H:%M:%S')}] {msg} (peak RAM {peak_gb:.1f} GB)", file=sys.stderr, flush=True)


def _build(data_dir, split, cfg, n_jobs, select=None, hook=None):
    """Read, normalise, block and featurise one split.

    Blocking and the context features always cover every Source 1 record, so
    they match what the model sees at test time. `select(s1_ids) -> bool mask`
    limits the expensive pair features (and the returned pairs) to some entities.
    `hook(keys, records, s1, pool)` runs once blocking is done (used for diagnostics).
    Returns s1 and pool (entity_id, country), all candidate pairs (i, j) and the featurised pairs.
    """
    s1_raw, pool_raw = read_sources(Path(data_dir) / split, split)
    n1, n2 = len(s1_raw), len(pool_raw)
    log(f"{split}: {n1} Source 1 records, {n2} Source 2/3 records")
    records = prepare(pd.concat([s1_raw, pool_raw], ignore_index=True), n_jobs)
    del s1_raw, pool_raw
    s1 = records[["entity_id", "country"]].iloc[:n1].reset_index(drop=True)
    pool = records[["entity_id", "country"]].iloc[n1:].reset_index(drop=True)
    log(f"{split}: normalised")

    keys = build_keys(records, n_jobs)
    records = records.drop(columns=["postcodes", "numbers"])  # now held as key matrices
    gc.collect()
    log(f"{split}: hashed blocking keys")
    i, j = generate_candidates(keys, n1, n2, cfg["k_name"], cfg["k_addr"], cfg["max_df"], cfg["fallback_df"],
                               n_jobs, log=log)
    log(f"{split}: {len(i)} candidate pairs ({len(i) / max(n1, 1):.1f} per Source 1 record)")
    cand = pd.DataFrame({"i": i, "j": j, **similarity_features(i, j, n1, keys, n_jobs)})
    del i, j
    for name, values in context_features(cand).items():
        cand[name] = values

    pairs = cand
    if select is not None:
        rows = np.flatnonzero(select(s1["entity_id"].to_numpy()))
    if hook is not None:
        hook(keys, records, s1, pool)
    if select is not None:
        pairs = cand[np.isin(cand["i"].to_numpy(), rows)].reset_index(drop=True)
        cand = cand[["i", "j"]]
    gc.collect()
    log(f"{split}: computing features for {len(pairs)} pairs")
    pairs = pair_features(pairs, records, keys, n1, n_jobs)
    del keys, records
    gc.collect()
    log(f"{split}: features done")
    return s1, pool, cand, pairs


def _has_id(ids, wanted):
    """Hash-based membership of string IDs (np.isin on object arrays is quadratic)."""
    return pd.Index(ids).isin(wanted)


def _gold_arrays(gold, s1, pool):
    """Gold pair keys (i * n_pool + j) and the gold list size of every Source 1 row."""
    s1_index = pd.Index(s1["entity_id"])
    pool_index = pd.Index(pool["entity_id"])
    owners = np.array([k for k, v in gold.items() for _ in v], dtype=object)
    members = np.array([m for v in gold.values() for m in v], dtype=object)
    gi = s1_index.get_indexer(owners) if len(owners) else np.zeros(0, np.int64)
    gj = pool_index.get_indexer(members) if len(members) else np.zeros(0, np.int64)
    ok = (gi >= 0) & (gj >= 0)
    gold_keys = np.unique(gi[ok].astype(np.int64) * len(pool) + gj[ok])
    gold_n = np.array([len(set(gold.get(e, ()))) for e in s1["entity_id"]], dtype=np.int64)
    return gold_keys, gold_n


def _labels(pairs, n_pool, gold_keys):
    keys = pairs["i"].to_numpy(np.int64) * n_pool + pairs["j"].to_numpy(np.int64)
    return np.isin(keys, gold_keys).astype(np.int8)


def _split_ids(s1_ids, val_frac, seed):
    ids = np.array(sorted(s1_ids))
    rng = np.random.default_rng(seed)
    n_val = int(round(len(ids) * val_frac))
    perm = rng.permutation(len(ids))
    return sorted(ids[perm[n_val:]].tolist()), sorted(ids[perm[:n_val]].tolist())


def _sample_ids(s1_ids, max_entities, seed):
    ids = np.array(sorted(s1_ids))
    if not max_entities or len(ids) <= max_entities:
        return ids.tolist()
    rng = np.random.default_rng(seed + 1)
    return sorted(ids[rng.choice(len(ids), size=max_entities, replace=False)].tolist())


def _val_report(pairs, prob, threshold, exclusive, s1, pool, gold, val_ids, cand):
    mask = matcher.select(pairs, prob, threshold, exclusive)
    preds = matcher.match_lists(pairs, mask, s1, pool)
    val_rows = np.flatnonzero(_has_id(s1["entity_id"], val_ids))
    in_val = np.isin(cand["i"].to_numpy(), val_rows)
    val_s1 = s1.iloc[val_rows]
    cands = candidate_lists(cand["i"].to_numpy()[in_val], cand["j"].to_numpy()[in_val], s1, pool)
    report = {"threshold": threshold, **macro_scores(preds, gold, val_ids),
              **blocking_scores(cands, gold, val_ids, len(pool))}
    countries = dict(zip(val_s1["entity_id"], val_s1["country"]))
    report["by_country"] = {
        c: macro_scores(preds, gold, [e for e in val_ids if countries[e] == c])
        for c in sorted(set(countries.values()))
    }
    return report


def _emit(report):
    print(json.dumps(report, indent=2))


def _blocking_config(args, config=None):
    keys = ("k_name", "k_addr", "max_df", "fallback_df")
    return {k: (config[k] if config else getattr(args, k)) for k in keys}


def cmd_train(args):
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = resolve_jobs(args.n_jobs)
    cfg = _blocking_config(args)
    split = {}

    def pick(s1_ids):
        sample = _sample_ids(s1_ids, args.max_train_entities, args.seed)
        split["train"], split["val"] = _split_ids(sample, args.val_frac, args.seed)
        return _has_id(s1_ids, sample)

    gold = read_id_lists(Path(args.data_dir) / "train" / "train_ground_truth.tsv")
    diagnosis = {}

    def diagnose_blocking(keys, records, s1, pool):
        rows = np.flatnonzero(_has_id(s1["entity_id"], split["val"]))
        gold_keys, _ = _gold_arrays(gold, s1, pool)
        report, examples = blocking_diagnose(keys, records, len(s1), len(pool), rows, gold_keys, cfg, n_jobs, log)
        examples.to_csv(model_dir / "blocking_misses.tsv", sep="\t", index=False)
        diagnosis.update(report)
        log(f"diagnose: {report['missed_pairs']} of {report['gold_pairs']} validation gold pairs missed; "
            f"{report['missed_sharing_no_key']} share no name or address key; "
            f"examples in {model_dir / 'blocking_misses.tsv'}")

    s1, pool, cand, pairs = _build(args.data_dir, "train", cfg, n_jobs, select=pick,
                                   hook=diagnose_blocking if args.diagnose_blocking and args.val_frac > 0 else None)
    train_ids, val_ids = split["train"], split["val"]
    gold_keys, gold_n = _gold_arrays(gold, s1, pool)
    labels = _labels(pairs, len(pool), gold_keys)

    owners = Counter(pid for ids in gold.values() for pid in ids)
    exclusive = all(n == 1 for n in owners.values())
    log(f"pool records shared between Source 1 entities in ground truth: "
        f"{sum(n > 1 for n in owners.values())} -> exclusive assignment {'on' if exclusive else 'off'}")

    report = {"exclusive": exclusive, "source1_entities": len(s1), "train_entities": len(train_ids),
              "val_entities": len(val_ids), "training_pairs": len(pairs), "positive_pairs": int(labels.sum())}
    if diagnosis:
        report["blocking_diagnosis"] = diagnosis
    threshold = 0.5
    if val_ids:
        val_rows = np.flatnonzero(_has_id(s1["entity_id"], val_ids))
        is_val = np.isin(pairs["i"].to_numpy(), val_rows)
        model_val = matcher.fit(pairs[~is_val], labels[~is_val], args.seed, args.model, args.device, log)
        log("fitted validation model")
        val_pairs = pairs[is_val].reset_index(drop=True)
        prob = matcher.score(model_val, val_pairs)
        pos = np.searchsorted(val_rows, val_pairs["i"].to_numpy())
        threshold, _ = matcher.tune_threshold(pos, val_pairs["j"].to_numpy(), prob, labels[is_val],
                                              gold_n[val_rows], exclusive)
        joblib.dump(model_val, model_dir / "model_val.joblib")
        report["validation"] = _val_report(val_pairs, prob, threshold, exclusive, s1, pool, gold, val_ids, cand)
        log(f"validation macro F0.5 = {report['validation']['f05']:.4f} at threshold {threshold:.2f}, "
            f"pair completeness {report['validation']['pair_completeness']:.4f}")
        del model_val, val_pairs, prob
    if args.threshold is not None:
        threshold = args.threshold
    report["threshold"] = threshold

    model = matcher.fit(pairs, labels, args.seed, args.model, args.device, log)
    joblib.dump(model, model_dir / "model.joblib")
    config = {
        "threshold": threshold, "exclusive": exclusive, **cfg, "seed": args.seed, "val_frac": args.val_frac,
        "max_train_entities": args.max_train_entities, "model": args.model, "report": report,
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))
    (model_dir / "split.json").write_text(json.dumps({"train": train_ids, "val": val_ids}))
    log(f"saved model, config and split to {model_dir}")
    _emit(report)


def _load_config(model_dir):
    path = Path(model_dir) / "config.json"
    if not path.exists():
        sys.exit(f"{path} not found: run `python -m src.cli train` first")
    return json.loads(path.read_text())


def cmd_evaluate(args):
    """Re-score the held-out split from scratch (train already stores this report in config.json)."""
    model_dir = Path(args.model_dir)
    config = _load_config(model_dir)
    val_ids = json.loads((model_dir / "split.json").read_text())["val"]
    if not val_ids:
        sys.exit("no validation split: retrain with --val-frac > 0")
    threshold = config["threshold"] if args.threshold is None else args.threshold
    s1, pool, cand, pairs = _build(args.data_dir, "train", _blocking_config(args, config), resolve_jobs(args.n_jobs),
                                   select=lambda ids: _has_id(ids, val_ids))
    gold = read_id_lists(Path(args.data_dir) / "train" / "train_ground_truth.tsv")
    prob = matcher.score(joblib.load(model_dir / "model_val.joblib"), pairs)
    _emit(_val_report(pairs, prob, threshold, config["exclusive"], s1, pool, gold, val_ids, cand))


def cmd_predict(args):
    config = _load_config(args.model_dir)
    threshold = config["threshold"] if args.threshold is None else args.threshold
    s1, pool, cand, pairs = _build(args.data_dir, "test", _blocking_config(args, config), resolve_jobs(args.n_jobs))
    prob = matcher.score(joblib.load(Path(args.model_dir) / "model.joblib"), pairs)
    mask = matcher.select(pairs, prob, threshold, config["exclusive"])
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    del pairs, cand, prob
    gc.collect()

    out_dir = Path(args.out_dir)
    s1_ids, pool_ids = s1["entity_id"].to_numpy(), pool["entity_id"].to_numpy()
    write_pair_lists(out_dir / "matching_results.tsv", MATCH_HEADER, s1_ids, i[mask], j[mask], pool_ids)
    write_pair_lists(out_dir / "candidate_pairs.tsv", CANDIDATE_HEADER, s1_ids, i, j, pool_ids)
    log(f"wrote {out_dir / 'matching_results.tsv'} and {out_dir / 'candidate_pairs.tsv'}")

    issues = check_outputs(out_dir, s1, pool)
    for issue in issues[:50]:
        log(f"CHECK FAILED: {issue}")
    matched_per_entity = np.bincount(i[mask], minlength=len(s1))
    _emit({
        "threshold": threshold,
        "entities": len(s1),
        "predicted_singletons": int((matched_per_entity == 0).sum()),
        "matched_pairs": int(mask.sum()),
        "candidate_pairs": len(i),
        "entities_by_country": {str(k): int(v) for k, v in s1["country"].value_counts().items()},
        "checks_passed": not issues,
    })
    if issues:
        sys.exit(1)


def cmd_run(args):
    cmd_train(args)
    gc.collect()
    cmd_predict(args)


def cmd_check(args):
    s1, pool = read_sources(Path(args.data_dir) / "test", "test")
    issues = check_outputs(args.out_dir, s1, pool)
    for n, issue in enumerate(issues, 1):
        print(f"{n}. {issue}")
    print("PASS" if not issues else "FAIL")
    sys.exit(1 if issues else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help_, *flags):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--data-dir", default="dataset", help="folder containing train/ and test/ (default: dataset)")
        if "model" in flags:
            p.add_argument("--model-dir", default="artifacts", help="fitted models, threshold and split (default: artifacts)")
            p.add_argument("--threshold", type=float, default=None, help="override the tuned decision threshold")
            p.add_argument("--n-jobs", type=int, default=0, help="worker processes/threads (default: 0 = all CPUs)")
        if "out" in flags:
            p.add_argument("--out-dir", default="output", help="where the two submission TSVs go (default: output)")
        if "train" in flags:
            p.add_argument("--val-frac", type=float, default=0.2, help="share of sampled train entities held out (default: 0.2)")
            p.add_argument("--seed", type=int, default=42, help="split and model seed (default: 42)")
            p.add_argument("--max-train-entities", type=int, default=400_000,
                           help="Source 1 train entities sampled for fitting and validation; 0 = all (default: 400000)")
            p.add_argument("--diagnose-blocking", action="store_true",
                           help="also re-block the validation entities with larger k/max-df and explain missed "
                                "gold pairs (report + blocking_misses.tsv in --model-dir)")
            p.add_argument("--model", choices=["hgb", "xgboost"], default="hgb",
                           help="classifier: scikit-learn HistGradientBoosting or XGBoost (default: hgb)")
            p.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                           help="where XGBoost trains; cuda uses the GPU, falling back to cpu (default: cpu)")
            p.add_argument("--k-name", type=int, default=15, help="name-key neighbours per entity (default: 15)")
            p.add_argument("--k-addr", type=int, default=10, help="address-key neighbours per entity (default: 10)")
            p.add_argument("--max-df", type=int, default=3000,
                           help="blocking uses keys found in at most this many pool records (default: 3000)")
            p.add_argument("--fallback-df", type=int, default=30000,
                           help="max pool frequency of the rarest-key fallback (default: 30000)")
        p.set_defaults(func=func)

    add("train", cmd_train, "fit the matcher, tune the F0.5 threshold on a held-out split", "model", "train")
    add("evaluate", cmd_evaluate, "score the held-out split: macro F0.5/P/R and blocking metrics", "model")
    add("predict", cmd_predict, "write matching_results.tsv and candidate_pairs.tsv for test", "model", "out")
    add("run", cmd_run, "train, then predict", "model", "out", "train")
    add("check", cmd_check, "check both output files against the submission rules", "out")

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
