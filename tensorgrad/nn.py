"""Neural-network modules, built only from ops in ops.py."""
import numpy as np

from . import ops
from .tensor import Tensor, no_grad


def Parameter(data, dtype=np.float32):
    return Tensor(np.asarray(data, dtype=dtype), requires_grad=True)


class Module:
    """Base class. Parameters and submodules are discovered from attributes
    (including lists of modules), in attribute-definition order."""

    def named_parameters(self, prefix=""):
        for name, value in vars(self).items():
            if isinstance(value, Tensor) and value.requires_grad:
                yield prefix + name, value
            elif isinstance(value, Module):
                yield from value.named_parameters(f"{prefix}{name}.")
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield from item.named_parameters(f"{prefix}{name}.{i}.")

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def num_parameters(self):
        return sum(p.size for p in self.parameters())

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def state_dict(self):
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state):
        params = dict(self.named_parameters())
        missing, unexpected = params.keys() - state.keys(), state.keys() - params.keys()
        if missing or unexpected:
            raise KeyError(f"missing={sorted(missing)} unexpected={sorted(unexpected)}")
        for name, p in params.items():
            if state[name].shape != p.shape:
                raise ValueError(f"{name}: shape {state[name].shape} != {p.shape}")
            p.data = np.asarray(state[name], dtype=p.dtype).copy()

    def save(self, path, **extra):
        """Save parameters (and any extra arrays) to an .npz file."""
        np.savez(path, **{f"param/{k}": v for k, v in self.state_dict().items()}, **extra)

    def load(self, path):
        with np.load(path) as f:
            self.load_state_dict({k[6:]: f[k] for k in f.files if k.startswith("param/")})

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    """y = x @ W + b, with W of shape (in, out)."""

    def __init__(self, n_in, n_out, bias=True, std=0.02, rng=None, dtype=np.float32):
        rng = rng or np.random.default_rng()
        self.weight = Parameter(rng.normal(0, std, (n_in, n_out)), dtype)
        self.bias = Parameter(np.zeros(n_out), dtype) if bias else None

    def forward(self, x):
        y = x @ self.weight
        return y + self.bias if self.bias is not None else y


class Embedding(Module):
    def __init__(self, num, dim, std=0.02, rng=None, dtype=np.float32):
        rng = rng or np.random.default_rng()
        self.weight = Parameter(rng.normal(0, std, (num, dim)), dtype)

    def forward(self, idx):
        return ops.embedding(self.weight, idx)


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5, dtype=np.float32):
        self.weight = Parameter(np.ones(dim), dtype)
        self.bias = Parameter(np.zeros(dim), dtype)
        self.eps = eps

    def forward(self, x):
        return ops.layer_norm(x, self.weight, self.bias, self.eps)


class CausalSelfAttention(Module):
    """Multi-head self-attention where position t only attends to positions <= t."""

    def __init__(self, d_model, n_head, rng=None, dtype=np.float32, proj_std=0.02):
        assert d_model % n_head == 0
        self.n_head = n_head
        self.qkv = Linear(d_model, 3 * d_model, rng=rng, dtype=dtype)
        self.proj = Linear(d_model, d_model, rng=rng, dtype=dtype, std=proj_std)

    def forward(self, x):
        B, T, C = x.shape
        H, hd = self.n_head, C // self.n_head
        qkv = self.qkv(x)  # (B, T, 3C)

        def heads(t):  # (B, T, C) -> (B, H, T, hd)
            return t.reshape(B, T, H, hd).transpose(0, 2, 1, 3)

        q, k, v = (heads(qkv[:, :, i * C:(i + 1) * C]) for i in range(3))
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(hd))  # (B, H, T, T)
        future = np.triu(np.ones((T, T), dtype=bool), k=1)  # True above the diagonal
        att = ops.softmax(ops.masked_fill(att, future, -np.inf), axis=-1)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(y)


class MLP(Module):
    def __init__(self, d_model, rng=None, dtype=np.float32, proj_std=0.02):
        self.fc = Linear(d_model, 4 * d_model, rng=rng, dtype=dtype)
        self.proj = Linear(4 * d_model, d_model, rng=rng, dtype=dtype, std=proj_std)

    def forward(self, x):
        return self.proj(ops.gelu(self.fc(x)))


class Block(Module):
    """Pre-LN transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, d_model, n_head, rng=None, dtype=np.float32, proj_std=0.02):
        self.ln1 = LayerNorm(d_model, dtype=dtype)
        self.attn = CausalSelfAttention(d_model, n_head, rng, dtype, proj_std)
        self.ln2 = LayerNorm(d_model, dtype=dtype)
        self.mlp = MLP(d_model, rng, dtype, proj_std)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(Module):
    def __init__(self, vocab_size, block_size, n_layer=4, n_head=4, d_model=128,
                 seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.block_size = block_size
        # GPT-2 trick: shrink residual-branch output projections by 1/sqrt(2 * n_layer)
        proj_std = 0.02 / np.sqrt(2 * n_layer)
        self.wte = Embedding(vocab_size, d_model, rng=rng, dtype=dtype)
        self.wpe = Embedding(block_size, d_model, rng=rng, dtype=dtype)
        self.blocks = [Block(d_model, n_head, rng, dtype, proj_std) for _ in range(n_layer)]
        self.ln_f = LayerNorm(d_model, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, bias=False, rng=rng, dtype=dtype)

    def forward(self, idx, targets=None):
        """idx: (B, T) int array -> logits (B, T, vocab) [, mean cross-entropy loss]."""
        idx = np.asarray(idx)
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"
        x = self.wte(idx) + self.wpe(np.arange(T))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits
        return logits, ops.cross_entropy(logits, targets)

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, rng=None):
        """Autoregressively sample tokens, appending to idx of shape (B, T)."""
        rng = rng or np.random.default_rng()
        idx = np.asarray(idx)
        with no_grad():
            for _ in range(max_new_tokens):
                logits = self(idx[:, -self.block_size:]).data[:, -1, :] / temperature
                if top_k is not None:
                    kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
                    logits = np.where(logits < kth, -np.inf, logits)
                probs = ops.softmax(Tensor(logits.astype(np.float64))).data
                nxt = np.array([rng.choice(len(p), p=p) for p in probs])
                idx = np.concatenate([idx, nxt[:, None]], axis=1)
        return idx
