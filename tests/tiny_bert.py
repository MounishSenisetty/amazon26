"""A tiny randomly initialised BERT saved locally, standing in for the pretrained model in tests."""

import string
from pathlib import Path


def make_tiny_bert(path):
    from transformers import BertConfig, BertModel, BertTokenizerFast

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    chars = list(string.ascii_lowercase + string.digits) + ["|"]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + ["##" + c for c in chars]
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    try:  # transformers >= 5 takes the vocabulary itself; older versions take the file
        tokenizer = BertTokenizerFast(vocab={t: k for k, t in enumerate(vocab)}, do_lower_case=True)
    except TypeError:
        tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(path)
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=256)
    BertModel(config).save_pretrained(path)
    return path
