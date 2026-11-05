"""Generate text from a trained checkpoint.

    python sample.py --prompt "ROMEO:" --tokens 500
"""
import argparse
import json

import numpy as np

from tensorgrad import nn
from tensorgrad.data import CharTokenizer


def load_checkpoint(path):
    with np.load(path) as f:
        config = json.loads(str(f["config"]))
        chars = str(f["chars"])
    model = nn.GPT(**config)
    model.load(path)
    return model, CharTokenizer(chars)


def generate(model, tok, prompt="\n", tokens=500, temperature=0.8, top_k=None, seed=0):
    idx = tok.encode(prompt)[None, :]
    out = model.generate(idx, tokens, temperature, top_k, np.random.default_rng(seed))
    return tok.decode(out[0])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ckpt", default="assets/model.npz")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="also write the sample to this file")
    args = p.parse_args(argv)
    model, tok = load_checkpoint(args.ckpt)
    text = generate(model, tok, args.prompt, args.tokens, args.temperature, args.top_k, args.seed)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    return text


if __name__ == "__main__":
    main()
