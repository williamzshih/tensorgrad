"""Gradchecks (and optional torch parity) for every primitive op."""
import numpy as np
import pytest

from tensorgrad import ops
from tensorgrad.tensor import Tensor, no_grad
from tests.gradcheck import gradcheck

try:
    import torch
except ImportError:  # parity tests skip without torch
    torch = None

rng = np.random.default_rng(42)
r = lambda *s: rng.standard_normal(s)
pos = lambda *s: rng.uniform(0.5, 2.0, s)
MASK = rng.random((3, 4)) > 0.5
W46 = r(4, 6)

# name: (tensorgrad fn, torch fn or None, input arrays)
# Many entries use operator overloads, so one lambda serves both libraries.
CASES = {
    "add": (lambda a, b: a + b, None, [r(3, 4), r(3, 4)]),
    "add_broadcast": (lambda a, b: a + b, None, [r(2, 3, 4), r(4)]),
    "add_broadcast_keepdim": (lambda a, b: a + b, None, [r(2, 3, 4), r(2, 1, 4)]),
    "add_scalar": (lambda a: a + 2.0 + (3.0 + a), None, [r(3)]),
    "sub_broadcast": (lambda a, b: a - b, None, [r(3, 1), r(1, 4)]),
    "rsub": (lambda a: 1.0 - a, None, [r(3)]),
    "mul_broadcast": (lambda a, b: a * b, None, [r(2, 3, 4), r(3, 1)]),
    "div_broadcast": (lambda a, b: a / b, None, [r(2, 3), pos(3)]),
    "rdiv": (lambda a: 2.0 / a, None, [pos(4)]),
    "neg": (lambda a: -a, None, [r(3, 2)]),
    "pow_const": (lambda a: a**3, None, [r(3, 4)]),
    "pow_frac": (lambda a: a**0.5, None, [pos(3, 4)]),
    "pow_tensor": (lambda a, b: a**b, None, [pos(3, 4), r(4)]),
    "matmul": (lambda a, b: a @ b, None, [r(3, 4), r(4, 5)]),
    "matmul_batched": (lambda a, b: a @ b, None, [r(2, 3, 4), r(2, 4, 5)]),
    "matmul_broadcast": (lambda a, b: a @ b, None, [r(2, 2, 3, 4), r(4, 5)]),
    "matmul_broadcast2": (lambda a, b: a @ b, None, [r(3, 4), r(2, 4, 5)]),
    "sum_all": (lambda a: a.sum(), None, [r(3, 4)]),
    "sum_axis": (lambda a: a.sum(axis=1), lambda a: a.sum(dim=1), [r(2, 3, 4)]),
    "sum_axes_keepdims": (lambda a: a.sum(axis=(0, 2), keepdims=True),
                          lambda a: a.sum(dim=(0, 2), keepdim=True), [r(2, 3, 4)]),
    "sum_neg_axis": (lambda a: a.sum(axis=-1), lambda a: a.sum(dim=-1), [r(2, 3, 4)]),
    "mean_all": (lambda a: a.mean(), None, [r(3, 4)]),
    "mean_axis_keepdims": (lambda a: a.mean(axis=-1, keepdims=True),
                           lambda a: a.mean(dim=-1, keepdim=True), [r(2, 3, 4)]),
    "max_all": (lambda a: a.max(), None, [r(3, 4)]),
    "max_axis": (lambda a: a.max(axis=1), lambda a: a.amax(dim=1), [r(2, 3, 4)]),
    "max_keepdims": (lambda a: a.max(axis=-1, keepdims=True),
                     lambda a: a.amax(dim=-1, keepdim=True), [r(2, 3, 4)]),
    "exp": (lambda a: a.exp(), None, [r(3, 4)]),
    "log": (lambda a: a.log(), None, [pos(3, 4)]),
    "tanh": (lambda a: a.tanh(), None, [r(3, 4)]),
    "relu": (lambda a: a.relu(), None, [r(3, 4)]),
    "gelu": (lambda a: a.gelu(),
             lambda a: torch.nn.functional.gelu(a, approximate="tanh"), [r(3, 4) * 2]),
    "reshape": (lambda a: a.reshape(4, 6) * Tensor(W46),
                lambda a: a.reshape(4, 6) * torch.tensor(W46), [r(2, 3, 4)]),
    "transpose": (lambda a: a.transpose(), lambda a: a.permute(2, 1, 0), [r(2, 3, 4)]),
    "permute": (lambda a: a.permute(1, 2, 0), lambda a: a.permute(1, 2, 0), [r(2, 3, 4)]),
    "swapaxes": (lambda a: ops.swapaxes(a, -1, -2), lambda a: a.transpose(-1, -2), [r(2, 3, 4)]),
    "getitem_slice": (lambda a: a[1:, ::2], None, [r(3, 4)]),
    "getitem_int": (lambda a: a[:, 2], None, [r(3, 4)]),
    "gather_repeated": (lambda a: a[np.array([[0, 2], [2, 2]])], None, [r(3, 4)]),
    "embedding": (lambda w: ops.embedding(w, np.array([[1, 0, 1], [3, 3, 3]])),
                  lambda w: torch.nn.functional.embedding(
                      torch.tensor([[1, 0, 1], [3, 3, 3]]), w), [r(4, 5)]),
    "concat": (lambda a, b: ops.concat([a, b, a], axis=1),
               lambda a, b: torch.cat([a, b, a], dim=1), [r(2, 3), r(2, 2)]),
    "masked_fill": (lambda a: ops.masked_fill(a, MASK, -5.0),
                    lambda a: a.masked_fill(torch.tensor(MASK), -5.0), [r(3, 4)]),
    "where": (lambda a, b: ops.where(MASK, a, b),
              lambda a, b: torch.where(torch.tensor(MASK), a, b), [r(3, 4), r(4)]),
    "reuse": (lambda a: a * a + a.exp() * a - a.sum(), None, [r(3)]),
    "diamond": (lambda a, b: (a @ b).tanh() @ b.transpose() + a,
                lambda a, b: (a @ b).tanh() @ b.T + a, [r(3, 4), r(4, 4)]),
}


@pytest.mark.parametrize("name", CASES)
def test_gradcheck(name):
    fn, _, arrays = CASES[name]
    gradcheck(fn, arrays)


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("name", CASES)
def test_torch_parity(name):
    fn, tfn, arrays = CASES[name]
    tfn = tfn or fn
    ours = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    theirs = [torch.tensor(a, requires_grad=True) for a in arrays]
    out, tout = fn(*ours), tfn(*theirs)
    np.testing.assert_allclose(out.data, tout.detach().numpy(), rtol=1e-10, atol=1e-12)
    w = rng.standard_normal(out.shape)
    out.backward(w)
    tout.backward(torch.tensor(w))
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a.grad, b.grad.numpy(), rtol=1e-10, atol=1e-12)


def test_unbroadcast():
    g = np.ones((2, 3, 4))
    assert ops.unbroadcast(g, (4,)).tolist() == [6.0] * 4
    assert ops.unbroadcast(g, (3, 1)).shape == (3, 1)
    assert np.all(ops.unbroadcast(g, (2, 1, 4)) == 3)
    assert ops.unbroadcast(g, ()).item() == 24


def test_grad_accumulates_across_uses_and_calls():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    (x * 3 + x).sum().backward()
    np.testing.assert_allclose(x.grad, [4, 4])
    (x * x).sum().backward()  # leaf grads accumulate like torch
    np.testing.assert_allclose(x.grad, [4 + 2, 4 + 4])


def test_max_ties_split_gradient():
    x = Tensor(np.array([[1.0, 3.0, 3.0]]), requires_grad=True)
    x.max(axis=1).sum().backward()
    np.testing.assert_allclose(x.grad, [[0, 0.5, 0.5]])


def test_no_grad_builds_no_graph():
    x = Tensor(np.ones(3), requires_grad=True)
    with no_grad():
        y = (x * 2).exp()
    assert not y.requires_grad and y._parents == ()


def test_constants_do_not_upcast_float32():
    x = Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
    y = (x * 2.0 + 1.0) / 3.0 - np.float64(1.0)
    assert y.dtype == np.float32
    y.sum().backward()
    assert x.grad.dtype == np.float32


def test_deep_graph():
    x = Tensor(np.ones(2), requires_grad=True)
    y = x
    for _ in range(3000):
        y = y + 1.0
    y.sum().backward()
    np.testing.assert_allclose(x.grad, [1, 1])


def test_backward_requires_scalar_or_grad():
    x = Tensor(np.ones(3), requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2).backward()
