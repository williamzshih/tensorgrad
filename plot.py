"""Plot the training log: python plot.py [--log assets/loss.csv] [--out assets/loss.png]"""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="assets/loss.csv")
    p.add_argument("--out", default="assets/loss.png")
    p.add_argument("--smooth", type=int, default=50, help="moving-average window for train loss")
    args = p.parse_args(argv)

    steps, train, val_steps, val = [], [], [], []
    with open(args.log) as f:
        for row in csv.DictReader(f):
            if row["train_loss"]:
                steps.append(int(row["step"]))
                train.append(float(row["train_loss"]))
            if row["val_loss"]:
                val_steps.append(int(row["step"]))
                val.append(float(row["val_loss"]))

    k = max(1, min(args.smooth, len(train)))
    smooth = [sum(train[max(0, i - k + 1):i + 1]) / len(train[max(0, i - k + 1):i + 1])
              for i in range(len(train))]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(steps, train, color="#9ecae1", lw=0.6, label="train (per step)")
    ax.plot(steps, smooth, color="#2171b5", lw=1.5, label=f"train ({k}-step avg)")
    ax.plot(val_steps, val, "o-", color="#d94801", ms=3, lw=1.5, label="validation")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (nats/char)")
    ax.set_title("tensorgrad char-GPT on Tiny Shakespeare")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
