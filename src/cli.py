"""Command-line entry point: python -m src.cli {train,evaluate,predict,run,check}."""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from . import matcher
from .blocking import candidate_lists, generate_candidates
from .checks import check_outputs
from .features import build_features
from .metrics import blocking_scores, macro_scores
from .normalize import prepare
from .tsv import CANDIDATE_HEADER, MATCH_HEADER, read_id_lists, read_sources, write_id_lists


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _build(data_dir, split, k_name, k_addr):
    s1, pool = read_sources(Path(data_dir) / split, split)
    log(f"{split}: {len(s1)} Source 1 records, {len(pool)} Source 2/3 records")
    s1, pool = prepare(s1), prepare(pool)
    cand = generate_candidates(s1, pool, k_name=k_name, k_addr=k_addr)
    log(f"{split}: {len(cand)} candidate pairs")
    pairs = build_features(cand, s1, pool)
    return s1, pool, cand, pairs


def _labels(pairs, s1, pool, gold):
    s1_ids = s1["entity_id"].to_numpy()[pairs["i"].to_numpy()]
    pool_ids = pool["entity_id"].to_numpy()[pairs["j"].to_numpy()]
    gold_sets = {k: set(v) for k, v in gold.items()}
    return np.array([p in gold_sets.get(s, ()) for s, p in zip(s1_ids, pool_ids)], dtype=np.int8)


def _split_ids(s1_ids, val_frac, seed):
    ids = np.array(sorted(s1_ids))
    rng = np.random.default_rng(seed)
    n_val = int(round(len(ids) * val_frac))
    perm = rng.permutation(len(ids))
    return sorted(ids[perm[n_val:]].tolist()), sorted(ids[perm[:n_val]].tolist())


def _emit(report):
    print(json.dumps(report, indent=2))


def cmd_train(args):
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    s1, pool, cand, pairs = _build(args.data_dir, "train", args.k_name, args.k_addr)
    gold = read_id_lists(Path(args.data_dir) / "train" / "train_ground_truth.tsv")
    labels = _labels(pairs, s1, pool, gold)

    owners = {}
    for s1_id, ids in gold.items():
        for pid in ids:
            owners[pid] = owners.get(pid, 0) + 1
    exclusive = all(n == 1 for n in owners.values())
    log(f"pool records shared between Source 1 entities in ground truth: "
        f"{sum(n > 1 for n in owners.values())} -> exclusive assignment {'on' if exclusive else 'off'}")

    train_ids, val_ids = _split_ids(s1["entity_id"], args.val_frac, args.seed)
    report = {"exclusive": exclusive, "train_entities": len(train_ids), "val_entities": len(val_ids)}
    threshold = 0.5
    if val_ids:
        pair_s1 = s1["entity_id"].to_numpy()[pairs["i"].to_numpy()]
        is_val = np.isin(pair_s1, val_ids)
        model_val = matcher.fit(pairs[~is_val], labels[~is_val], args.seed)
        prob = matcher.score(model_val, pairs[is_val])
        threshold, val_scores = matcher.tune_threshold(
            pairs[is_val], prob, s1, pool, gold, val_ids, exclusive)
        joblib.dump(model_val, model_dir / "model_val.joblib")
        report["validation"] = {
            "threshold": threshold,
            **val_scores,
            **blocking_scores(candidate_lists(cand, s1, pool), gold, val_ids, len(pool)),
        }
        log(f"validation macro F0.5 = {val_scores['f05']:.4f} at threshold {threshold:.2f}")
    if args.threshold is not None:
        threshold = args.threshold
    report["threshold"] = threshold

    model = matcher.fit(pairs, labels, args.seed)
    joblib.dump(model, model_dir / "model.joblib")
    config = {
        "threshold": threshold, "exclusive": exclusive, "k_name": args.k_name,
        "k_addr": args.k_addr, "seed": args.seed, "val_frac": args.val_frac,
        "report": report,
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
    model_dir = Path(args.model_dir)
    config = _load_config(model_dir)
    val_ids = json.loads((model_dir / "split.json").read_text())["val"]
    if not val_ids:
        sys.exit("no validation split: retrain with --val-frac > 0")
    s1, pool, cand, pairs = _build(args.data_dir, "train", config["k_name"], config["k_addr"])
    gold = read_id_lists(Path(args.data_dir) / "train" / "train_ground_truth.tsv")
    threshold = config["threshold"] if args.threshold is None else args.threshold

    is_val = np.isin(s1["entity_id"].to_numpy()[pairs["i"].to_numpy()], val_ids)
    val_pairs = pairs[is_val]
    prob = matcher.score(joblib.load(model_dir / "model_val.joblib"), val_pairs)
    mask = matcher.select(val_pairs, prob, threshold, config["exclusive"])
    preds = matcher.match_lists(val_pairs, mask, s1, pool)

    report = {"threshold": threshold, **macro_scores(preds, gold, val_ids),
              **blocking_scores(candidate_lists(cand, s1, pool), gold, val_ids, len(pool))}
    countries = dict(zip(s1["entity_id"], s1["country"]))
    report["by_country"] = {
        c: macro_scores(preds, gold, [e for e in val_ids if countries[e] == c])
        for c in sorted({countries[e] for e in val_ids})
    }
    _emit(report)


def cmd_predict(args):
    config = _load_config(args.model_dir)
    threshold = config["threshold"] if args.threshold is None else args.threshold
    s1, pool, cand, pairs = _build(args.data_dir, "test", config["k_name"], config["k_addr"])
    prob = matcher.score(joblib.load(Path(args.model_dir) / "model.joblib"), pairs)
    mask = matcher.select(pairs, prob, threshold, config["exclusive"])

    out_dir = Path(args.out_dir)
    order = s1["entity_id"].tolist()
    matches = matcher.match_lists(pairs, mask, s1, pool)
    write_id_lists(out_dir / "matching_results.tsv", MATCH_HEADER, order, matches)
    write_id_lists(out_dir / "candidate_pairs.tsv", CANDIDATE_HEADER, order, candidate_lists(cand, s1, pool))
    log(f"wrote {out_dir / 'matching_results.tsv'} and {out_dir / 'candidate_pairs.tsv'}")

    issues = check_outputs(out_dir, s1, pool)
    for issue in issues:
        log(f"CHECK FAILED: {issue}")
    _emit({
        "threshold": threshold,
        "entities": len(order),
        "predicted_singletons": sum(not v for v in matches.values()),
        "matched_pairs": int(mask.sum()),
        "candidate_pairs": len(cand),
        "entities_by_country": s1["country"].value_counts().to_dict(),
        "checks_passed": not issues,
    })
    if issues:
        sys.exit(1)


def cmd_run(args):
    cmd_train(args)
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
        if "out" in flags:
            p.add_argument("--out-dir", default="output", help="where the two submission TSVs go (default: output)")
        if "train" in flags:
            p.add_argument("--val-frac", type=float, default=0.2, help="share of Source 1 train entities held out (default: 0.2)")
            p.add_argument("--seed", type=int, default=42, help="split and model seed (default: 42)")
            p.add_argument("--k-name", type=int, default=15, help="name-similarity neighbours per entity (default: 15)")
            p.add_argument("--k-addr", type=int, default=10, help="address-similarity neighbours per entity (default: 10)")
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
