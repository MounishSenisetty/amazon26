"""Optional transformer components (enabled with --neural; best on a GPU).

- BiEncoder: a small sentence-embedding transformer fine-tuned on training
  gold pairs with in-batch negatives. Its embeddings give a third blocking
  channel (nearest neighbours by cosine) and an emb_cos feature for every pair.
- CrossEncoder: the same kind of backbone with a one-logit head, fine-tuned
  to read a (Source 1, pool) record pair together and score it.

Both start from pretrained weights (DEFAULT_MODEL: Apache-2.0, 22.7M
parameters) and are fine-tuned only on the challenge training data. Records
are fed as their normalised text "name | address".
"""

import math
import time

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _torch():
    import torch
    return torch


def pick_device():
    torch = _torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def record_texts(records):
    return (records["name_norm"].astype(object) + " | " + records["addr_norm"].astype(object)).to_numpy(object)


def _length_order(*text_arrays):
    lengths = sum(np.fromiter((len(t) for t in texts), dtype=np.int64, count=len(texts)) for texts in text_arrays)
    return np.argsort(lengths, kind="stable")


def _parallel(module, device):
    torch = _torch()
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        return torch.nn.DataParallel(module)
    return module


def _autocast(device):
    torch = _torch()
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def _grad_scaler(device):
    torch = _torch()
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        return amp.GradScaler(enabled=device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def _schedule(optimizer, total_steps):
    torch = _torch()
    warmup = max(1, int(0.1 * total_steps))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (total_steps - s) / max(1, total_steps - warmup)))


class _Pooler:
    """Mean-pooled, L2-normalised sentence embeddings from a transformer encoder."""

    @staticmethod
    def build(backbone):
        torch = _torch()

        class Pool(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, attention_mask):
                hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                return torch.nn.functional.normalize(pooled.float(), dim=-1)

        return Pool(backbone)


class BiEncoder:
    def __init__(self, path, device, max_len=64):
        from transformers import AutoModel, AutoTokenizer
        self.device, self.max_len = device, max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.pool = _Pooler.build(AutoModel.from_pretrained(path)).to(device)

    def _encode_batch(self, module, texts):
        enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        return module(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))

    def fit(self, left, right, epochs=1, batch=256, lr=5e-5, seed=0, log=print):
        """In-batch-negatives (MultipleNegativesRanking) loss over (left[k], right[k]) positive pairs.

        Each left text should occur once, so no batch holds two positives of one entity.
        """
        torch = _torch()
        n = len(left)
        if n < 2:
            return self
        torch.manual_seed(seed)
        module = _parallel(self.pool, self.device)
        module.train()
        steps = epochs * math.ceil(n / batch)
        opt = torch.optim.AdamW(self.pool.parameters(), lr=lr)
        sched = _schedule(opt, steps)
        scaler = _grad_scaler(self.device)
        rng = np.random.default_rng(seed)
        t0, step = time.time(), 0
        for _ in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                if len(idx) < 2:
                    continue
                with _autocast(self.device):
                    a = self._encode_batch(module, left[idx])
                    b = self._encode_batch(module, right[idx])
                    sims = a @ b.T * 20.0
                    target = torch.arange(len(idx), device=self.device)
                    loss = (torch.nn.functional.cross_entropy(sims, target)
                            + torch.nn.functional.cross_entropy(sims.T, target)) / 2
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                if step % 200 == 0 or step == steps:
                    log(f"bi-encoder step {step}/{steps}: loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
        self.pool.eval()
        return self

    def encode(self, texts, batch=1024, out_device=None):
        """Embeddings of `texts`, float16 on CUDA (float32 on CPU), on out_device (default: self.device)."""
        torch = _torch()
        module = _parallel(self.pool, self.device)
        module.eval()
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        out_device = out_device or self.device
        dim = self.pool.model.config.hidden_size
        out = torch.empty((len(texts), dim), dtype=dtype, device=out_device)
        order = _length_order(texts)
        with torch.inference_mode(), _autocast(self.device):
            for s in range(0, len(texts), batch):
                idx = order[s:s + batch]
                out[torch.as_tensor(idx, device=out_device)] = self._encode_batch(module, texts[idx]).to(out_device, dtype)
        return out

    def save(self, path):
        self.pool.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def top_k(queries, keys, k, query_batch=1024, key_block=500_000):
    """For each row of `queries`, the k rows of `keys` with the largest dot product.

    queries may live on the CPU; keys should live on the compute device.
    Returns (query_row, key_row) int32 numpy arrays.
    """
    torch = _torch()
    k = min(k, keys.shape[0])
    if k <= 0 or queries.shape[0] == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    rows, cols = [], []
    with torch.inference_mode():
        for s in range(0, queries.shape[0], query_batch):
            q = queries[s:s + query_batch].to(keys.device, keys.dtype)
            best_v = best_i = None
            for b in range(0, keys.shape[0], key_block):
                sims = q @ keys[b:b + key_block].T
                v, i = sims.topk(min(k, sims.shape[1]), dim=1)
                i = i + b
                if best_v is not None:
                    v, pos = torch.cat([best_v, v], 1).topk(k, dim=1)
                    i = torch.cat([best_i, i], 1).gather(1, pos)
                best_v, best_i = v, i
            rows.append(np.repeat(np.arange(s, s + q.shape[0], dtype=np.int32), best_i.shape[1]))
            cols.append(best_i.reshape(-1).cpu().numpy().astype(np.int32))
    return np.concatenate(rows), np.concatenate(cols)


def pair_cosine(queries, keys, i, j, chunk=1_000_000):
    """Row-wise dot products queries[i] . keys[j] (embeddings are L2-normalised, so cosine)."""
    torch = _torch()
    out = np.empty(len(i), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, len(i), chunk):
            ii = torch.as_tensor(np.asarray(i[s:s + chunk], dtype=np.int64), device=queries.device)
            jj = torch.as_tensor(np.asarray(j[s:s + chunk], dtype=np.int64), device=keys.device)
            q = queries[ii].to(keys.device, torch.float32)
            out[s:s + len(ii)] = (q * keys[jj].float()).sum(1).cpu().numpy()
    return out


class CrossEncoder:
    def __init__(self, path, device, max_len=128):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device, self.max_len = device, max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1).to(device)

    def _logits(self, module, left, right):
        enc = self.tokenizer(list(left), list(right), padding=True, truncation=True, max_length=self.max_len,
                             return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = module(**enc)
        return (out.logits if hasattr(out, "logits") else out[0]).squeeze(-1).float()

    def fit(self, left, right, labels, epochs=1, batch=128, lr=3e-5, seed=0, log=print):
        torch = _torch()
        n = len(labels)
        if n < 2 or len(np.unique(labels)) < 2:
            log(f"cross-encoder: {n} training pairs with a single class; keeping the pretrained head untrained")
            return self
        torch.manual_seed(seed)
        module = _parallel(self.model, self.device)
        module.train()
        steps = epochs * math.ceil(n / batch)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        sched = _schedule(opt, steps)
        scaler = _grad_scaler(self.device)
        y_all = torch.as_tensor(np.asarray(labels, dtype=np.float32))
        rng = np.random.default_rng(seed)
        t0, step = time.time(), 0
        for _ in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                with _autocast(self.device):
                    logits = self._logits(module, left[idx], right[idx])
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, y_all[idx].to(self.device))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                if step % 500 == 0 or step == steps:
                    log(f"cross-encoder step {step}/{steps}: loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
        self.model.eval()
        return self

    def predict(self, left, right, batch=1024, log=print):
        """Match probability of each (left[k], right[k]) pair."""
        torch = _torch()
        module = _parallel(self.model, self.device)
        module.eval()
        out = np.empty(len(left), dtype=np.float32)
        order = _length_order(left, right)
        t0 = time.time()
        with torch.inference_mode(), _autocast(self.device):
            for n_done, s in enumerate(range(0, len(left), batch)):
                idx = order[s:s + batch]
                out[idx] = torch.sigmoid(self._logits(module, left[idx], right[idx])).cpu().numpy()
                if n_done and n_done % 2000 == 0:
                    log(f"cross-encoder scored {s + len(idx)}/{len(left)} pairs ({time.time() - t0:.0f}s)")
        return out

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
