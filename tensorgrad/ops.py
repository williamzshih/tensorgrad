"""Primitive ops and their hand-written gradients.

Convention: every op computes its forward value with NumPy and returns
Tensor._make(value, parents, backward, name), where backward(g) maps the
upstream gradient g = dL/d(out) to a tuple (dL/d(parent) for each parent).
"""
import numpy as np

from .tensor import Tensor


def as_tensor(x, like=None):
    """Wrap constants as Tensors, matching `like`'s dtype (avoids upcasting)."""
    if isinstance(x, Tensor):
        return x
    dtype = like.dtype if isinstance(like, Tensor) else None
    return Tensor(np.asarray(x, dtype=dtype))


def _pair(a, b):
    a = as_tensor(a, b)
    return a, as_tensor(b, a)


def unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, undoing NumPy broadcasting.

    Broadcasting (1) prepends size-1 axes and (2) stretches size-1 axes.
    A value copied to many outputs receives the sum of their gradients, so
    we sum over the prepended axes, then over the stretched ones.
    """
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    stretched = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad.reshape(shape)


# --- elementwise binary ------------------------------------------------------
def add(a, b):
    a, b = _pair(a, b)
    return Tensor._make(a.data + b.data, (a, b),
                        lambda g: (unbroadcast(g, a.shape), unbroadcast(g, b.shape)), "add")


def sub(a, b):
    a, b = _pair(a, b)
    return Tensor._make(a.data - b.data, (a, b),
                        lambda g: (unbroadcast(g, a.shape), unbroadcast(-g, b.shape)), "sub")


def mul(a, b):
    a, b = _pair(a, b)
    return Tensor._make(a.data * b.data, (a, b),
                        lambda g: (unbroadcast(g * b.data, a.shape),
                                   unbroadcast(g * a.data, b.shape)), "mul")


def div(a, b):
    a, b = _pair(a, b)
    return Tensor._make(a.data / b.data, (a, b),
                        lambda g: (unbroadcast(g / b.data, a.shape),
                                   unbroadcast(-g * a.data / b.data**2, b.shape)), "div")


def neg(a):
    return Tensor._make(-a.data, (a,), lambda g: (-g,), "neg")


def pow(a, b):
    """a ** b. If b is a Python number it is treated as a constant."""
    if not isinstance(b, Tensor) and np.isscalar(b):
        a = as_tensor(a)
        return Tensor._make(a.data**b, (a,),
                            lambda g: (g * b * a.data ** (b - 1),), "pow")
    a, b = _pair(a, b)
    out = a.data**b.data

    def backward(g):
        ga = unbroadcast(g * b.data * a.data ** (b.data - 1), a.shape)
        gb = None
        if b.requires_grad:  # d/db a^b = a^b log a  (needs a > 0)
            gb = unbroadcast(g * out * np.log(a.data), b.shape)
        return ga, gb

    return Tensor._make(out, (a, b), backward, "pow")


def matmul(a, b):
    """Batched matmul with broadcasting over leading dims (both ndim >= 2)."""
    a, b = _pair(a, b)
    assert a.ndim >= 2 and b.ndim >= 2, "matmul expects ndim >= 2"
    # out = a @ b  =>  dL/da = g @ b^T,  dL/db = a^T @ g  (then unbroadcast batch dims)
    return Tensor._make(a.data @ b.data, (a, b),
                        lambda g: (unbroadcast(g @ b.data.swapaxes(-1, -2), a.shape),
                                   unbroadcast(a.data.swapaxes(-1, -2) @ g, b.shape)),
                        "matmul")


# --- elementwise unary -------------------------------------------------------
def exp(a):
    out = np.exp(a.data)
    return Tensor._make(out, (a,), lambda g: (g * out,), "exp")


def log(a):
    return Tensor._make(np.log(a.data), (a,), lambda g: (g / a.data,), "log")


def tanh(a):
    out = np.tanh(a.data)
    return Tensor._make(out, (a,), lambda g: (g * (1 - out * out),), "tanh")


def relu(a):
    return Tensor._make(np.maximum(a.data, 0), (a,), lambda g: (g * (a.data > 0),), "relu")


_GELU_C = float(np.sqrt(2 / np.pi))  # Python float: keeps float32 as float32


def gelu(a):
    """GELU, tanh approximation (as in GPT-2):
    0.5 x (1 + tanh(c (x + 0.044715 x^3))),  c = sqrt(2/pi)."""
    x = a.data
    t = np.tanh(_GELU_C * (x + 0.044715 * x**3))
    out = 0.5 * x * (1 + t)

    def backward(g):
        dt = (1 - t * t) * _GELU_C * (1 + 3 * 0.044715 * x * x)
        return (g * (0.5 * (1 + t) + 0.5 * x * dt),)

    return Tensor._make(out, (a,), backward, "gelu")


# --- reductions --------------------------------------------------------------
def _expand(g, shape, axis, keepdims):
    """Re-insert reduced axes so g broadcasts against the input shape."""
    if axis is not None and not keepdims:
        axes = (axis,) if np.isscalar(axis) else axis
        g = np.expand_dims(g, tuple(ax % len(shape) for ax in axes))
    return g


def sum(a, axis=None, keepdims=False):
    return Tensor._make(a.data.sum(axis=axis, keepdims=keepdims), (a,),
                        lambda g: (np.broadcast_to(_expand(g, a.shape, axis, keepdims),
                                                   a.shape).copy(),), "sum")


def mean(a, axis=None, keepdims=False):
    n = a.size / np.asarray(a.data.sum(axis=axis, keepdims=keepdims)).size
    return sum(a, axis, keepdims) / n


def max(a, axis=None, keepdims=False):
    """Max over axes. The gradient goes to the argmax; ties share it equally."""
    out = a.data.max(axis=axis, keepdims=True)

    def backward(g):
        mask = a.data == out
        share = mask / mask.sum(axis=axis, keepdims=True)
        return (share * _expand(g, a.shape, axis, keepdims),)

    value = out if keepdims else out.squeeze(axis=axis) if axis is not None else out.reshape(())
    return Tensor._make(value, (a,), backward, "max")


# --- shape ops ---------------------------------------------------------------
def reshape(a, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor._make(a.data.reshape(shape), (a,),
                        lambda g: (g.reshape(a.shape),), "reshape")


def transpose(a, *axes):
    """Permute axes (default: reverse them, like np.transpose)."""
    if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
        axes = tuple(axes[0])
    axes = axes or tuple(reversed(range(a.ndim)))
    inverse = np.argsort(axes)
    return Tensor._make(a.data.transpose(axes), (a,),
                        lambda g: (g.transpose(inverse),), "transpose")


permute = transpose


def swapaxes(a, i, j):
    axes = list(range(a.ndim))
    axes[i], axes[j] = axes[j], axes[i]
    return transpose(a, axes)


def getitem(a, idx):
    """Indexing / slicing / gathering. Integer-array indices may repeat
    (e.g. an embedding lookup with repeated tokens), so the backward uses
    np.add.at, which accumulates instead of overwriting."""
    if isinstance(idx, Tensor):
        idx = idx.data.astype(np.int64)

    def backward(g):
        grad = np.zeros_like(a.data)
        np.add.at(grad, idx, g)
        return (grad,)

    return Tensor._make(a.data[idx], (a,), backward, "getitem")


def embedding(weight, idx):
    """Row lookup: weight[idx] for an integer array idx of any shape."""
    return getitem(weight, np.asarray(idx))


def concat(tensors, axis=0):
    tensors = [as_tensor(t) for t in tensors]
    sizes = [t.shape[axis] for t in tensors]
    splits = np.cumsum(sizes)[:-1]
    return Tensor._make(np.concatenate([t.data for t in tensors], axis=axis), tuple(tensors),
                        lambda g: tuple(np.split(g, splits, axis=axis)), "concat")


def masked_fill(a, mask, value):
    """Replace entries where mask is True with `value` (no gradient there)."""
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), a.shape)
    return Tensor._make(np.where(mask, np.asarray(value, a.dtype), a.data), (a,),
                        lambda g: (np.where(mask, 0, g).astype(g.dtype),), "masked_fill")


def where(cond, a, b):
    """Elementwise cond ? a : b, with broadcasting."""
    cond = np.asarray(cond, dtype=bool)
    a, b = _pair(a, b)
    return Tensor._make(np.where(cond, a.data, b.data), (a, b),
                        lambda g: (unbroadcast(np.where(cond, g, 0).astype(g.dtype), a.shape),
                                   unbroadcast(np.where(cond, 0, g).astype(g.dtype), b.shape)),
                        "where")


# --- fused, numerically stable ops (hand-derived backward) -------------------
def softmax(a, axis=-1):
    """exp(x - max) / sum(exp(x - max)). Subtracting the max avoids overflow
    and does not change the result."""
    e = np.exp(a.data - a.data.max(axis=axis, keepdims=True))
    s = e / e.sum(axis=axis, keepdims=True)
    # dL/dx = s * (g - sum(g * s))
    return Tensor._make(s, (a,),
                        lambda g: (s * (g - (g * s).sum(axis=axis, keepdims=True)),),
                        "softmax")


def log_softmax(a, axis=-1):
    """x - max - log(sum(exp(x - max)))."""
    shifted = a.data - a.data.max(axis=axis, keepdims=True)
    out = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    s = np.exp(out)
    # dL/dx = g - softmax(x) * sum(g)
    return Tensor._make(out, (a,),
                        lambda g: (g - s * g.sum(axis=axis, keepdims=True),),
                        "log_softmax")


def cross_entropy(logits, targets):
    """Mean negative log-likelihood of integer `targets` under softmax(logits).

    logits: (..., C) Tensor;  targets: integer array of shape (...).
    Fused backward: dL/dlogits = (softmax(logits) - one_hot(targets)) / N.
    """
    targets = np.asarray(targets.data if isinstance(targets, Tensor) else targets)
    C = logits.shape[-1]
    z = logits.data.reshape(-1, C)
    t = targets.reshape(-1).astype(np.int64)
    N = z.shape[0]
    shifted = z - z.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -logp[np.arange(N), t].mean()

    def backward(g):
        d = np.exp(logp)
        d[np.arange(N), t] -= 1
        return ((d * (g / N)).reshape(logits.shape),)

    return Tensor._make(np.asarray(loss, dtype=logits.dtype), (logits,), backward,
                        "cross_entropy")


def layer_norm(x, weight, bias, eps=1e-5):
    """Normalize over the last axis, then scale and shift:
    y = (x - mean) / sqrt(var + eps) * weight + bias   (biased variance)."""
    D = x.shape[-1]
    mu = x.data.mean(axis=-1, keepdims=True)
    xc = x.data - mu
    rstd = 1.0 / np.sqrt((xc * xc).mean(axis=-1, keepdims=True) + eps)
    xhat = xc * rstd
    out = xhat * weight.data + bias.data

    def backward(g):
        lead = tuple(range(g.ndim - 1))
        dweight = (g * xhat).sum(axis=lead)
        dbias = g.sum(axis=lead)
        dxhat = g * weight.data
        # dx = rstd/D * (D*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
        dx = rstd / D * (D * dxhat - dxhat.sum(axis=-1, keepdims=True)
                         - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True))
        return dx, dweight, dbias

    return Tensor._make(out, (x, weight, bias), backward, "layer_norm")
