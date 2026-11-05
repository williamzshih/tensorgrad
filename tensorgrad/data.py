"""Character-level tokenizer, dataset download and random batching."""
import os
import urllib.request

import numpy as np

TINY_SHAKESPEARE_URLS = [
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    # fallback: public-domain complete works from Project Gutenberg
    "https://www.gutenberg.org/cache/epub/100/pg100.txt",
]


def load_text(path="data/input.txt", urls=TINY_SHAKESPEARE_URLS):
    """Read `path`, downloading it first if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        for url in urls:
            try:
                print(f"downloading {url} -> {path}")
                urllib.request.urlretrieve(url, path)
                break
            except OSError as e:
                print(f"  failed: {e}")
        else:
            raise RuntimeError(f"could not download a dataset; put a text file at {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return np.array([self.stoi[c] for c in text], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.chars[int(i)] for i in ids)


def train_val_split(ids, val_fraction=0.1):
    n = int(len(ids) * (1 - val_fraction))
    return ids[:n], ids[n:]


def get_batch(ids, batch_size, block_size, rng):
    """Random windows: x = ids[i:i+T], y = ids[i+1:i+T+1] (next-char targets)."""
    starts = rng.integers(0, len(ids) - block_size, batch_size)
    x = np.stack([ids[s:s + block_size] for s in starts])
    y = np.stack([ids[s + 1:s + block_size + 1] for s in starts])
    return x, y
