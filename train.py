"""Train a character-level GPT on Tiny Shakespeare with tensorgrad.

    python train.py                 # default config (~25 min on 4 CPU cores)
    python train.py --steps 200     # quick smoke run
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from tensorgrad import nn
from tensorgrad.data import CharTokenizer, get_batch, load_text, train_val_split
from tensorgrad.optim import AdamW, clip_grad_norm, cosine_lr
from tensorgrad.tensor import no_grad


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--out", default="assets/model.npz", help="checkpoint path")
    p.add_argument("--log", default="assets/loss.csv", help="loss log (CSV)")
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args(argv)


def estimate_loss(model, ids, args, rng):
    with no_grad():
        losses = [model(*get_batch(ids, args.batch_size, args.block_size, rng))[1].item()
                  for _ in range(args.eval_batches)]
    return float(np.mean(losses))


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    text = load_text(args.data)
    tok = CharTokenizer(text)
    train_ids, val_ids = train_val_split(tok.encode(text))
    config = dict(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                  n_head=args.n_head, d_model=args.d_model, seed=args.seed)
    model = nn.GPT(**config)
    print(f"{len(text):,} chars, vocab {tok.vocab_size}, {model.num_parameters():,} parameters")

    # No weight decay on biases and LayerNorm parameters (the 1-D tensors).
    params = model.parameters()
    opt = AdamW([{"params": [p for p in params if p.ndim >= 2]},
                 {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
                lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    for path in (args.out, args.log):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    eval_rng = np.random.default_rng(args.seed + 1)
    start = time.time()
    with open(args.log, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["step", "lr", "train_loss", "val_loss", "elapsed_s"])
        for step in range(args.steps + 1):
            val_loss = ""
            if step % args.eval_every == 0 or step == args.steps:
                val_loss = estimate_loss(model, val_ids, args, eval_rng)
                print(f"step {step:5d} | val {val_loss:.4f} | {time.time() - start:7.1f}s")
            if step == args.steps:
                log.writerow([step, "", "", f"{val_loss:.4f}", f"{time.time() - start:.1f}"])
                break
            opt.lr = cosine_lr(step, args.lr, args.steps, args.warmup, args.min_lr)
            x, y = get_batch(train_ids, args.batch_size, args.block_size, rng)
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            clip_grad_norm(params, args.grad_clip)
            opt.step()
            val_str = f"{val_loss:.4f}" if val_loss != "" else ""
            log.writerow([step, f"{opt.lr:.6g}", f"{loss.item():.4f}", val_str,
                          f"{time.time() - start:.1f}"])
            if step % 50 == 0:
                f.flush()
                print(f"step {step:5d} | train {loss.item():.4f} | lr {opt.lr:.2e}")

    model.save(args.out, config=np.array(json.dumps(config)), chars=np.array("".join(tok.chars)))
    print(f"saved {args.out} after {time.time() - start:.0f}s")
    return model, tok


if __name__ == "__main__":
    main()
