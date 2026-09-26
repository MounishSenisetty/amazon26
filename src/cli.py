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

from . import matcher, neural
from .blocking import build_keys, candidate_lists, generate_candidates
from .blocking import _union
from .blocking import diagnose as blocking_diagnose
from .checks import check_outputs
from .features import NEURAL_FEATURES, STAGE2_FEATURES, context_features, pair_features, similarity_features
from .metrics import blocking_scores, macro_scores
from .normalize import prepare
from .parallel import resolve_jobs
from .tsv import CANDIDATE_HEADER, MATCH_HEADER, read_id_lists, read_sources, write_pair_lists


def log(msg):
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    print(f"[{time.strftime('%H:%M:%S')}] {msg} (peak RAM {peak_gb:.1f} GB)", file=sys.stderr, flush=True)


def _build(data_dir, split, cfg, n_jobs, select=None, hook=None, encoder=None):
    """Read, normalise, block and featurise one split.

    Blocking and the context features always cover every Source 1 record, so
    they match what the model sees at test time. `select(s1_ids) -> bool mask`
    limits the expensive pair features (and the returned pairs) to some entities.
    `hook(keys, records, s1, pool)` runs once blocking is done (used for diagnostics).
    `encoder(texts, s1, pool) -> neural.BiEncoder` turns on the embedding channel
    and the emb_cos/emb_rank features (--neural).
    Returns s1 and pool (entity_id, country), all candidate pairs (i, j), the
    featurised pairs, and the records' "name | address" texts (None without an encoder).
    """
    s1_raw, pool_raw = read_sources(Path(data_dir) / split, split)
    n1, n2 = len(s1_raw), len(pool_raw)
    log(f"{split}: {n1} Source 1 records, {n2} Source 2/3 records")
    records = prepare(pd.concat([s1_raw, pool_raw], ignore_index=True), n_jobs)
    del s1_raw, pool_raw
    s1 = records[["entity_id", "country"]].iloc[:n1].reset_index(drop=True)
    pool = records[["entity_id", "country"]].iloc[n1:].reset_index(drop=True)
    rows = np.flatnonzero(select(s1["entity_id"].to_numpy())) if select is not None else None
    log(f"{split}: normalised")

    keys = build_keys(records, n_jobs)
    records = records.drop(columns=["postcodes", "numbers"])  # now held as key matrices
    gc.collect()
    log(f"{split}: hashed blocking keys")
    i, j = generate_candidates(keys, n1, n2, cfg["k_name"], cfg["k_addr"], cfg["max_df"], cfg["fallback_df"],
                               n_jobs, log=log)
    texts, emb_cos = None, None
    if encoder is not None:
        texts = neural.record_texts(records)
        i, j, emb_cos = _embedding_channel(encoder(texts, s1, pool), texts, n1, i, j, cfg["k_emb"], split)
    log(f"{split}: {len(i)} candidate pairs ({len(i) / max(n1, 1):.1f} per Source 1 record)")
    cand = pd.DataFrame({"i": i, "j": j, **similarity_features(i, j, n1, keys, n_jobs)})
    if emb_cos is not None:
        cand["emb_cos"] = emb_cos
    del i, j, emb_cos
    for name, values in context_features(cand).items():
        cand[name] = values

    pairs = cand
    if hook is not None:
        hook(keys, records, s1, pool)
    if rows is not None:
        pairs = cand[np.isin(cand["i"].to_numpy(), rows)].reset_index(drop=True)
        cand = cand[["i", "j"]]
    gc.collect()
    log(f"{split}: computing features for {len(pairs)} pairs")
    pairs = pair_features(pairs, records, keys, n1, n_jobs)
    del keys, records
    gc.collect()
    log(f"{split}: features done")
    return s1, pool, cand, pairs, texts


def _embedding_channel(enc, texts, n1, i, j, k_emb, split):
    """Add each Source 1 record's k_emb nearest pool records by bi-encoder cosine; return pairs and emb_cos."""
    import torch

    t = time.time()
    pool_emb = enc.encode(texts[n1:])
    s1_emb = enc.encode(texts[:n1], out_device=torch.device("cpu"))
    log(f"{split}: encoded {len(texts)} records with the bi-encoder ({time.time() - t:.0f}s)")
    ie, je = neural.top_k(s1_emb, pool_emb, k_emb)
    n2 = len(texts) - n1
    before = len(i)
    i, j = _union([(i, j), (ie, je)], n2)
    log(f"blocking: embedding channel proposed {len(ie)} pairs, {len(i) - before} of them new")
    emb_cos = neural.pair_cosine(s1_emb, pool_emb, i, j)
    del pool_emb, s1_emb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return i, j, emb_cos


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
    source = config if config else vars(args)
    cfg = {k: source[k] for k in ("k_name", "k_addr", "max_df", "fallback_df")}
    cfg["k_emb"] = source.get("k_emb", 10)
    return cfg


def _partition(s1_ids, seed, max_gbdt, max_ce, max_bi):
    """Disjoint seeded entity sets for --neural: GBDT sample (incl. validation), cross-encoder, bi-encoder."""
    ids = np.array(sorted(s1_ids))
    n = len(ids)
    perm = np.random.default_rng(seed + 1).permutation(n)
    n_ce, n_bi = min(max_ce, int(0.2 * n)), min(max_bi, int(0.2 * n))
    n_gbdt = min(max_gbdt or n, n - n_ce - n_bi)
    parts = np.split(perm, [n_gbdt, n_gbdt + n_ce, n_gbdt + n_ce + n_bi])[:3]
    return [sorted(ids[p].tolist()) for p in parts]


def cmd_train(args):
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = resolve_jobs(args.n_jobs)
    cfg = _blocking_config(args)
    split = {"ce": [], "bi": []}

    def pick(s1_ids):
        if args.neural:
            sample, split["ce"], split["bi"] = _partition(s1_ids, args.seed, args.max_train_entities,
                                                          args.ce_train_entities, args.bienc_train_entities)
        else:
            sample = _sample_ids(s1_ids, args.max_train_entities, args.seed)
        split["train"], split["val"] = _split_ids(sample, args.val_frac, args.seed)
        return _has_id(s1_ids, sample + split["ce"])

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

    def fit_encoder(texts, s1, pool):
        """Fine-tune the bi-encoder on one gold pair per bi-encoder entity (disjoint from the GBDT sample)."""
        device = neural.pick_device()
        log(f"neural: device {device}; loading {args.neural_model}")
        enc = neural.BiEncoder(args.neural_model, device)
        gold_keys, _ = _gold_arrays(gold, s1, pool)
        n1, n2 = len(s1), len(pool)
        rows = np.flatnonzero(_has_id(s1["entity_id"], split["bi"]))
        gk = gold_keys[np.isin(gold_keys // n2, rows)]
        gk = gk[np.random.default_rng(args.seed).permutation(len(gk))]
        gk = gk[np.unique(gk // n2, return_index=True)[1]]
        log(f"neural: fine-tuning the bi-encoder on {len(gk)} gold pairs")
        enc.fit(texts[gk // n2], texts[n1 + gk % n2], epochs=args.neural_epochs, batch=256, lr=5e-5,
                seed=args.seed, log=log)
        enc.save(model_dir / "bi_encoder")
        return enc

    s1, pool, cand, pairs, texts = _build(
        args.data_dir, "train", cfg, n_jobs, select=pick,
        hook=diagnose_blocking if args.diagnose_blocking and args.val_frac > 0 else None,
        encoder=fit_encoder if args.neural else None)
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
    config = {"exclusive": exclusive, **cfg, "seed": args.seed, "val_frac": args.val_frac,
              "max_train_entities": args.max_train_entities, "model": args.model}
    if args.neural:
        report.update(ce_entities=len(split["ce"]), bienc_entities=len(split["bi"]))
        threshold, config["neural"] = _fit_neural(args, pairs, labels, texts, s1, pool, gold, gold_n, split, cand,
                                                  exclusive, model_dir, report)
    else:
        threshold = _fit_single(args, pairs, labels, s1, pool, gold, gold_n, val_ids, cand, exclusive, model_dir,
                                report)
    if args.threshold is not None:
        threshold = args.threshold
    report["threshold"] = threshold
    config.update(threshold=threshold, report=report)
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))
    (model_dir / "split.json").write_text(json.dumps({"train": train_ids, "val": val_ids}))
    log(f"saved models, config and split to {model_dir}")
    _emit(report)


def _tune_on_val(val_pairs, prob, labels, val_rows, gold_n, exclusive):
    pos = np.searchsorted(val_rows, val_pairs["i"].to_numpy())
    threshold, _ = matcher.tune_threshold(pos, val_pairs["j"].to_numpy(), prob, labels, gold_n[val_rows], exclusive)
    return threshold


def _fit_single(args, pairs, labels, s1, pool, gold, gold_n, val_ids, cand, exclusive, model_dir, report):
    """One classifier over FEATURES: tune the threshold on validation, then refit on the whole sample."""
    threshold = 0.5
    if val_ids:
        val_rows = np.flatnonzero(_has_id(s1["entity_id"], val_ids))
        is_val = np.isin(pairs["i"].to_numpy(), val_rows)
        model_val = matcher.fit(pairs[~is_val], labels[~is_val], args.seed, args.model, args.device, log)
        log("fitted validation model")
        val_pairs = pairs[is_val].reset_index(drop=True)
        prob = matcher.score(model_val, val_pairs)
        threshold = _tune_on_val(val_pairs, prob, labels[is_val], val_rows, gold_n, exclusive)
        joblib.dump(model_val, model_dir / "model_val.joblib")
        report["validation"] = _val_report(val_pairs, prob, threshold, exclusive, s1, pool, gold, val_ids, cand)
        log(f"validation macro F0.5 = {report['validation']['f05']:.4f} at threshold {threshold:.2f}, "
            f"pair completeness {report['validation']['pair_completeness']:.4f}")
    joblib.dump(matcher.fit(pairs, labels, args.seed, args.model, args.device, log), model_dir / "model.joblib")
    return threshold


def _fit_neural(args, pairs, labels, texts, s1, pool, gold, gold_n, split, cand, exclusive, model_dir, report):
    """Two-stage cascade with a cross-encoder (--neural). Returns (threshold, config entries).

    Stage 1: the GBDT over NEURAL_FEATURES. Out-of-fold probabilities on the
    sample pick a prefilter threshold tau that keeps --prefilter-recall of true
    pairs. The cross-encoder is fine-tuned on the separate cross-encoder
    entities' pairs that pass tau, then scores the sample's surviving pairs.
    Stage 2: a GBDT over STAGE2_FEATURES (adds p1 and ce_score) on survivors.
    """
    n1 = len(s1)
    rows_of = lambda ids: np.flatnonzero(_has_id(s1["entity_id"], ids))  # noqa: E731
    val_rows, ce_rows = rows_of(split["val"]), rows_of(split["ce"])
    ent = pairs["i"].to_numpy()
    is_gbdt = np.isin(ent, rows_of(split["train"] + split["val"]))
    is_ce, is_val = np.isin(ent, ce_rows), np.isin(ent, val_rows)

    def fit(mask, frame, y, features):
        return matcher.fit(frame[mask], y[mask], args.seed, args.model, args.device, log, features)

    fold = np.random.default_rng(args.seed + 2).integers(0, 2, n1)[ent]
    p1 = np.zeros(len(pairs), dtype=np.float32)
    for f in (0, 1):
        model = fit(is_gbdt & (fold != f), pairs, labels, NEURAL_FEATURES)
        held = is_gbdt & (fold == f)
        p1[held] = matcher.score(model, pairs[held], NEURAL_FEATURES)
    model1 = fit(is_gbdt, pairs, labels, NEURAL_FEATURES)
    p1[is_ce] = matcher.score(model1, pairs[is_ce], NEURAL_FEATURES)
    joblib.dump(model1, model_dir / "model_stage1.joblib")
    positives = labels.astype(bool) & is_gbdt
    tau = float(np.clip(np.quantile(p1[positives], 1 - args.prefilter_recall), 1e-4, 0.05)) if positives.any() else 0.01
    kept = is_gbdt & (p1 >= tau)
    log(f"stage 1: prefilter p1 >= {tau:.4g} keeps {kept.sum()} of {is_gbdt.sum()} sample pairs and "
        f"{(kept & positives).sum() / max(positives.sum(), 1):.4f} of true pairs")
    if is_val.any():
        vp = pairs[is_val].reset_index(drop=True)
        t1 = _tune_on_val(vp, p1[is_val], labels[is_val], val_rows, gold_n, exclusive)
        report["validation_stage1"] = _val_report(vp, p1[is_val], t1, exclusive, s1, pool, gold, split["val"], cand)
        log(f"stage 1 (no cross-encoder) validation macro F0.5 = {report['validation_stage1']['f05']:.4f}, "
            f"pair completeness {report['validation_stage1']['pair_completeness']:.4f}")

    device = neural.pick_device()
    ce_idx = _cross_encoder_sample(is_ce, p1, labels, tau, args.ce_train_pairs, args.seed)
    j_all = pairs["j"].to_numpy()
    log(f"neural: fine-tuning the cross-encoder on {len(ce_idx)} pairs ({labels[ce_idx].mean():.2%} positive)")
    ce = neural.CrossEncoder(args.neural_model, device)
    ce.fit(texts[ent[ce_idx]], texts[n1 + j_all[ce_idx]], labels[ce_idx], epochs=args.neural_epochs, batch=128,
           lr=3e-5, seed=args.seed, log=log)
    ce.save(model_dir / "cross_encoder")

    stage2 = pairs[kept].reset_index(drop=True)
    stage2["p1"] = p1[kept]
    log(f"neural: cross-encoder scoring {len(stage2)} pairs")
    stage2["ce_score"] = ce.predict(texts[stage2["i"].to_numpy()], texts[n1 + stage2["j"].to_numpy()], log=log)
    y2 = labels[kept]
    v2 = np.isin(stage2["i"].to_numpy(), val_rows)
    threshold = 0.5
    if v2.any() and (~v2).any():
        model_val = fit(~v2, stage2, y2, STAGE2_FEATURES)
        vp = stage2[v2].reset_index(drop=True)
        prob = matcher.score(model_val, vp, STAGE2_FEATURES)
        threshold = _tune_on_val(vp, prob, y2[v2], val_rows, gold_n, exclusive)
        report["validation"] = _val_report(vp, prob, threshold, exclusive, s1, pool, gold, split["val"], cand)
        log(f"validation macro F0.5 = {report['validation']['f05']:.4f} at threshold {threshold:.2f} "
            f"(stage 1 alone: {report.get('validation_stage1', {}).get('f05', float('nan')):.4f})")
    joblib.dump(fit(np.ones(len(stage2), dtype=bool), stage2, y2, STAGE2_FEATURES), model_dir / "model_stage2.joblib")
    return threshold, {"tau": tau, "k_emb": args.k_emb, "prefilter_recall": args.prefilter_recall,
                       "base_model": args.neural_model}


def _cross_encoder_sample(is_ce, p1, labels, tau, cap, seed):
    """Cross-encoder training pairs: all positives, hard negatives (p1 >= tau/5) and some easier negatives.

    The easier negatives guarantee both classes even when stage 1 already separates most pairs.
    """
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(is_ce & (labels == 1))
    hard = np.flatnonzero(is_ce & (labels == 0) & (p1 >= tau / 5))
    easy = np.flatnonzero(is_ce & (labels == 0) & (p1 < tau / 5))
    n_easy = min(len(easy), max(len(hard), len(pos) // 2))
    idx = np.concatenate([pos, hard, rng.choice(easy, n_easy, replace=False)])
    if len(idx) > cap:
        idx = rng.choice(idx, cap, replace=False)
    return np.sort(idx)


def _load_config(model_dir):
    path = Path(model_dir) / "config.json"
    if not path.exists():
        sys.exit(f"{path} not found: run `python -m src.cli train` first")
    return json.loads(path.read_text())


def cmd_evaluate(args):
    """Re-score the held-out split from scratch (train already stores this report in config.json)."""
    model_dir = Path(args.model_dir)
    config = _load_config(model_dir)
    if config.get("neural"):
        sys.exit("evaluate does not re-run --neural models; train stored the validation report in config.json")
    val_ids = json.loads((model_dir / "split.json").read_text())["val"]
    if not val_ids:
        sys.exit("no validation split: retrain with --val-frac > 0")
    threshold = config["threshold"] if args.threshold is None else args.threshold
    s1, pool, cand, pairs, _ = _build(args.data_dir, "train", _blocking_config(args, config),
                                      resolve_jobs(args.n_jobs), select=lambda ids: _has_id(ids, val_ids))
    gold = read_id_lists(Path(args.data_dir) / "train" / "train_ground_truth.tsv")
    prob = matcher.score(joblib.load(model_dir / "model_val.joblib"), pairs)
    _emit(_val_report(pairs, prob, threshold, config["exclusive"], s1, pool, gold, val_ids, cand))


def _neural_scores(model_dir, conf, pairs, texts, n1):
    """Stage 1 -> prefilter -> cross-encoder -> stage 2 probabilities for every pair (0 if pruned)."""
    p1 = matcher.score(joblib.load(model_dir / "model_stage1.joblib"), pairs, NEURAL_FEATURES)
    kept = p1 >= conf["tau"]
    stage2 = pairs[kept].reset_index(drop=True)
    stage2["p1"] = p1[kept]
    log(f"stage 1: {kept.sum()} of {len(pairs)} pairs pass the prefilter; cross-encoder scoring them")
    ce = neural.CrossEncoder(str(model_dir / "cross_encoder"), neural.pick_device())
    stage2["ce_score"] = ce.predict(texts[stage2["i"].to_numpy()], texts[n1 + stage2["j"].to_numpy()], log=log)
    prob = np.zeros(len(pairs), dtype=np.float32)
    prob[kept] = matcher.score(joblib.load(model_dir / "model_stage2.joblib"), stage2, STAGE2_FEATURES)
    return prob


def cmd_predict(args):
    model_dir = Path(args.model_dir)
    config = _load_config(model_dir)
    threshold = config["threshold"] if args.threshold is None else args.threshold
    conf = config.get("neural")

    def load_encoder(texts, s1, pool):
        return neural.BiEncoder(str(model_dir / "bi_encoder"), neural.pick_device())

    s1, pool, cand, pairs, texts = _build(args.data_dir, "test", _blocking_config(args, config),
                                          resolve_jobs(args.n_jobs), encoder=load_encoder if conf else None)
    if conf:
        prob = _neural_scores(model_dir, conf, pairs, texts, len(s1))
    else:
        prob = matcher.score(joblib.load(model_dir / "model.joblib"), pairs)
    mask = matcher.select(pairs, prob, threshold, config["exclusive"])
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    del pairs, cand, prob, texts
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
            p.add_argument("--neural", action="store_true",
                           help="add the transformer stages: fine-tuned bi-encoder (embedding blocking channel and "
                                "emb_cos feature) and a fine-tuned cross-encoder re-scorer; needs a GPU in practice")
            p.add_argument("--neural-model", default=neural.DEFAULT_MODEL,
                           help=f"pretrained transformer (Hugging Face id or local path) (default: {neural.DEFAULT_MODEL})")
            p.add_argument("--k-emb", type=int, default=10, help="embedding neighbours per entity (default: 10)")
            p.add_argument("--bienc-train-entities", type=int, default=500_000,
                           help="Source 1 entities reserved to fine-tune the bi-encoder (default: 500000)")
            p.add_argument("--ce-train-entities", type=int, default=300_000,
                           help="Source 1 entities reserved to fine-tune the cross-encoder (default: 300000)")
            p.add_argument("--ce-train-pairs", type=int, default=1_000_000,
                           help="max cross-encoder training pairs (default: 1000000)")
            p.add_argument("--neural-epochs", type=int, default=1, help="fine-tuning epochs (default: 1)")
            p.add_argument("--prefilter-recall", type=float, default=0.998,
                           help="share of true pairs the stage-1 prefilter keeps for the cross-encoder (default: 0.998)")
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
