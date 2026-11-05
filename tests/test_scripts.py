"""Smoke test: train.py -> sample.py -> plot.py on a tiny config."""
import importlib.util
import pathlib

import numpy as np

from tensorgrad.data import CharTokenizer, get_batch, train_val_split

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tokenizer_and_batches():
    tok = CharTokenizer("hello world")
    assert tok.decode(tok.encode("hello")) == "hello" and tok.vocab_size == 8
    ids = np.arange(100)
    tr, va = train_val_split(ids)
    assert len(tr) == 90 and len(va) == 10
    x, y = get_batch(tr, 4, 8, np.random.default_rng(0))
    assert x.shape == y.shape == (4, 8)
    np.testing.assert_array_equal(y[:, :-1], x[:, 1:])


def test_train_sample_plot(tmp_path):
    data = tmp_path / "text.txt"
    data.write_text("to be or not to be, that is the question.\n" * 50)
    ckpt, log = tmp_path / "m.npz", tmp_path / "loss.csv"
    load_script("train").main(["--data", str(data), "--out", str(ckpt), "--log", str(log),
                               "--steps", "30", "--n-layer", "1", "--d-model", "16",
                               "--n-head", "2", "--block-size", "16", "--batch-size", "8",
                               "--eval-every", "10", "--eval-batches", "2", "--warmup", "5"])
    rows = log.read_text().splitlines()
    assert rows[0] == "step,lr,train_loss,val_loss,elapsed_s" and len(rows) == 32
    text = load_script("sample").main(["--ckpt", str(ckpt), "--prompt", "to ", "--tokens", "20"])
    assert text.startswith("to ") and len(text) == 23
    load_script("plot").main(["--log", str(log), "--out", str(tmp_path / "loss.png")])
    assert (tmp_path / "loss.png").stat().st_size > 0
