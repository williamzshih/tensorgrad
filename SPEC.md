# SPEC: tensorgrad — a transformer trained on an autodiff engine written from scratch

## Goal
Build a reverse-mode autodiff engine in Python + NumPy (no PyTorch/JAX in the library code), then use it to train a small character-level GPT. It's a portfolio project, so correctness, test coverage and a clear README matter more than speed or features.

## Hard rules
- Library code may only import `numpy` and the standard library. `torch` is allowed **only in tests** as a reference implementation, and those tests skip when torch isn't installed.
- Every op's backward pass is checked against central finite differences (float64, relative tolerance ~1e-5) before moving on.
- Work milestone by milestone. At the end of each one: run the full test suite, fix failures, and commit with a clear message. Don't start the next milestone while tests are red.
- Keep it small and readable. Prefer clarity over cleverness, and keep each file under ~400 lines.

## Layout
```
tensorgrad/
  tensor.py      # Tensor class, graph, backward(), topo sort
  ops.py         # primitive ops + their gradients
  nn.py          # Module, Linear, Embedding, LayerNorm, Attention, MLP, Block, GPT
  optim.py       # SGD, AdamW, grad clipping
  data.py        # char tokenizer + batching
train.py         # training script (CLI args, logs loss to CSV)
sample.py        # load checkpoint, generate text
plot.py          # loss curve -> assets/loss.png
tests/
README.md
```

## Milestones

**M1: Scalar engine (warm-up, ~30 min).** A `Value` class with +, *, pow, exp, log, tanh, relu and backward via topological sort. Tests: gradient checks; train a 2-layer MLP on a toy 2D classification set to >95% accuracy.

**M2: Tensor engine.** A `Tensor` wrapping `np.ndarray` with `.grad` and backward. Ops: add, sub, mul, div, neg, pow, matmul (batched), sum/mean over axes (keepdims), max, exp, log, tanh, gelu, reshape, transpose/permute, indexing/gather (for embeddings), concat, masked_fill, where.
- Broadcasting must be correct in backward: reduce grads back to the input's shape (`unbroadcast` helper).
- Gradients accumulate when a tensor is used more than once.
- Tests: a finite-difference gradcheck for every op, including broadcast cases and reused tensors. If torch is available, also compare forward and backward against torch.

**M3: Fused, numerically stable ops.** `log_softmax` / `softmax` (max-subtraction trick) and `cross_entropy(logits, targets)` with a hand-derived fused backward (`softmax - one_hot`, divided by N). `layer_norm` with a hand-derived backward. Gradcheck both. Also check that the fused backward matches the gradient you get from composing primitives.

**M4: NN modules.** `Module` base with `parameters()`, `zero_grad()`, `state_dict()`/save/load (npz). Linear, Embedding, LayerNorm, causal multi-head self-attention (with a causal mask), MLP (4x, GELU), pre-LN transformer Block with residuals, GPT (token + learned positional embeddings, N blocks, final LN, LM head). Tests: shape tests; gradcheck a tiny GPT end-to-end (vocab 5, d=8, 1 layer, seq 4); a causality test (changing a future token must not change earlier logits).

**M5: Optimizer + training.** SGD, AdamW (decoupled weight decay, bias correction), global-norm grad clipping, cosine LR with warmup. Tests: AdamW matches torch's for a few steps on a fixed problem (if torch is present). Overfit test: the model must memorize a single short batch to near-zero loss.

**M6: Train on real data.** Character-level Tiny Shakespeare (or any public-domain text if the download fails). Start with a config that trains in ~15–30 min on a laptop CPU, e.g. 4 layers, 4 heads, d=128, context 64. Log train/val loss to CSV. Write `plot.py` to produce `assets/loss.png` and save a sample generation to `assets/sample.txt`.

**M7: README.** What it is, a quick start, architecture overview, results (loss plot, sample text, training time and hardware), how correctness is verified (gradcheck + torch parity), and limitations. Add a section **"Derivations"** with headings for: backprop through broadcasting, softmax + cross-entropy, LayerNorm backward, attention backward. **Leave the derivation bodies as `TODO(Will)`. Do not write them.**

## Will's part (not Claude's)
Will writes the Derivations section by hand and checks each one against the code in `ops.py`. If his derivation and the code disagree, the code is wrong until proven otherwise.

## Definition of done
All tests pass; `python train.py` reproduces the loss curve; `python sample.py` prints plausible Shakespeare-ish text; the README renders cleanly on GitHub with the plot and sample.