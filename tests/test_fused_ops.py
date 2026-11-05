"""M3: fused softmax / log_softmax / cross_entropy / layer_norm."""
import numpy as np
import pytest

from tensorgrad import ops
from tensorgrad.tensor import Tensor
from tests.gradcheck import gradcheck

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

rng = np.random.default_rng(7)
TARGETS = np.array([[0, 3, 1], [4, 4, 2]])


# Reference implementations composed from primitive ops only.
def composed_log_softmax(x, axis=-1):
    m = Tensor(x.data.max(axis=axis, keepdims=True))  # constant shift
    s = x - m
    return s - s.exp().sum(axis=axis, keepdims=True).log()


def composed_softmax(x, axis=-1):
    return composed_log_softmax(x, axis).exp()


def composed_cross_entropy(logits, targets):
    C = logits.shape[-1]
    logp = composed_log_softmax(logits.reshape(-1, C))
    t = targets.reshape(-1)
    return -(logp[np.arange(len(t)), t]).mean()


def composed_layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    return xc / (var + eps) ** 0.5 * w + b


FUSED = {
    "softmax": (lambda x: ops.softmax(x), composed_softmax, [rng.standard_normal((2, 3, 5))]),
    "softmax_axis0": (lambda x: ops.softmax(x, axis=0), lambda x: composed_softmax(x, 0),
                      [rng.standard_normal((4, 3))]),
    "log_softmax": (lambda x: ops.log_softmax(x), composed_log_softmax,
                    [rng.standard_normal((2, 3, 5))]),
    "cross_entropy": (lambda x: ops.cross_entropy(x, TARGETS),
                      lambda x: composed_cross_entropy(x, TARGETS),
                      [rng.standard_normal((2, 3, 5))]),
    "layer_norm": (ops.layer_norm, composed_layer_norm,
                   [rng.standard_normal((2, 3, 6)), rng.standard_normal(6),
                    rng.standard_normal(6)]),
}


@pytest.mark.parametrize("name", FUSED)
def test_gradcheck(name):
    fused, _, arrays = FUSED[name]
    gradcheck(fused, arrays)


@pytest.mark.parametrize("name", FUSED)
def test_fused_matches_composed(name):
    fused, composed, arrays = FUSED[name]
    results = []
    for fn in (fused, composed):
        ts = [Tensor(a.copy(), requires_grad=True) for a in arrays]
        out = fn(*ts)
        w = np.random.default_rng(1).standard_normal(out.shape)
        out.backward(w)
        results.append((out.data, [t.grad for t in ts]))
    (o1, g1), (o2, g2) = results
    np.testing.assert_allclose(o1, o2, rtol=1e-10, atol=1e-12)
    for a, b in zip(g1, g2):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def test_stable_for_huge_logits():
    x = Tensor(np.array([[1000.0, 0.0, -1000.0]]), requires_grad=True)
    assert np.all(np.isfinite(ops.log_softmax(x).data))
    np.testing.assert_allclose(ops.softmax(x).data, [[1, 0, 0]], atol=1e-12)
    loss = ops.cross_entropy(x, np.array([2]))
    assert loss.item() == pytest.approx(2000.0)
    loss.backward()
    np.testing.assert_allclose(x.grad, [[1, 0, -1]], atol=1e-12)


def test_cross_entropy_uniform():
    loss = ops.cross_entropy(Tensor(np.zeros((4, 10))), np.arange(4))
    assert loss.item() == pytest.approx(np.log(10))


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_torch_parity():
    x = rng.standard_normal((2, 3, 5))
    w, b = rng.standard_normal(5), rng.standard_normal(5)
    cases = [
        (lambda t: ops.softmax(t[0]), lambda t: F.softmax(t[0], dim=-1), [x]),
        (lambda t: ops.log_softmax(t[0]), lambda t: F.log_softmax(t[0], dim=-1), [x]),
        (lambda t: ops.cross_entropy(t[0], TARGETS),
         lambda t: F.cross_entropy(t[0].reshape(-1, 5), torch.tensor(TARGETS).reshape(-1)), [x]),
        (lambda t: ops.layer_norm(*t),
         lambda t: F.layer_norm(t[0], (5,), t[1], t[2], eps=1e-5), [x, w, b]),
    ]
    for ours_fn, torch_fn, arrays in cases:
        ours = [Tensor(a.copy(), requires_grad=True) for a in arrays]
        theirs = [torch.tensor(a, requires_grad=True) for a in arrays]
        out, tout = ours_fn(ours), torch_fn(theirs)
        np.testing.assert_allclose(out.data, tout.detach().numpy(), rtol=1e-9, atol=1e-12)
        g = rng.standard_normal(out.shape)
        out.backward(g)
        tout.backward(torch.tensor(g))
        for a, t in zip(ours, theirs):
            np.testing.assert_allclose(a.grad, t.grad.numpy(), rtol=1e-9, atol=1e-11)
