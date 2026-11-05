"""M5: optimizers, clipping, LR schedule, and an overfit test."""
import numpy as np
import pytest

from tensorgrad import nn, ops
from tensorgrad.optim import SGD, AdamW, clip_grad_norm, cosine_lr
from tensorgrad.tensor import Tensor

try:
    import torch
except ImportError:
    torch = None

rng = np.random.default_rng(0)
X, Y = rng.standard_normal((16, 4)), rng.standard_normal((16, 3))
W0, B0 = rng.standard_normal((4, 3)), rng.standard_normal(3)


def run_ours(make_opt, steps):
    w, b = Tensor(W0.copy(), requires_grad=True), Tensor(B0.copy(), requires_grad=True)
    opt = make_opt([w, b])
    for _ in range(steps):
        opt.zero_grad()
        loss = (((Tensor(X) @ w + b) - Tensor(Y)) ** 2).mean()
        loss.backward()
        opt.step()
    return w.data, b.data


def run_torch(make_opt, steps):
    w, b = torch.tensor(W0, requires_grad=True), torch.tensor(B0, requires_grad=True)
    opt = make_opt([w, b])
    for _ in range(steps):
        opt.zero_grad()
        loss = (((torch.tensor(X) @ w + b) - torch.tensor(Y)) ** 2).mean()
        loss.backward()
        opt.step()
    return w.detach().numpy(), b.detach().numpy()


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("ours,theirs", [
    (lambda p: AdamW(p, lr=0.05, betas=(0.9, 0.95), weight_decay=0.1),
     lambda p: torch.optim.AdamW(p, lr=0.05, betas=(0.9, 0.95), weight_decay=0.1)),
    (lambda p: AdamW(p, lr=0.01, weight_decay=0.0),
     lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.0)),
    (lambda p: SGD(p, lr=0.05, momentum=0.9, weight_decay=0.01),
     lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=0.01)),
])
def test_matches_torch(ours, theirs):
    for a, b in zip(run_ours(ours, 10), run_torch(theirs, 10)):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_adamw_first_step_by_hand():
    # With bias correction the first Adam step moves each weight by ~lr * sign(grad).
    p = Tensor(np.array([1.0, -2.0]), requires_grad=True)
    p.grad = np.array([0.3, -4.0])
    AdamW([p], lr=0.1, weight_decay=0.5).step()
    expected = np.array([1.0, -2.0]) * (1 - 0.1 * 0.5) - 0.1 * np.sign([0.3, -4.0])
    np.testing.assert_allclose(p.data, expected, rtol=1e-6)


def test_param_groups():
    a = Tensor(np.ones(2), requires_grad=True)
    b = Tensor(np.ones(2), requires_grad=True)
    a.grad, b.grad = np.zeros(2), np.zeros(2)
    opt = AdamW([{"params": [a]}, {"params": [b], "weight_decay": 0.0}], lr=0.1, weight_decay=0.5)
    opt.step()
    np.testing.assert_allclose(a.data, 0.95)
    np.testing.assert_allclose(b.data, 1.0)
    opt.lr = 0.01
    assert all(g["lr"] == 0.01 for g in opt.param_groups)


def test_clip_grad_norm():
    a = Tensor(np.zeros(2), requires_grad=True)
    b = Tensor(np.zeros(1), requires_grad=True)
    a.grad, b.grad = np.array([3.0, 0.0]), np.array([4.0])
    assert clip_grad_norm([a, b], 1.0) == pytest.approx(5.0)
    np.testing.assert_allclose(np.concatenate([a.grad, b.grad]), [0.6, 0, 0.8], rtol=1e-5)
    assert clip_grad_norm([a, b], 10.0) == pytest.approx(1.0, rel=1e-5)  # no-op
    np.testing.assert_allclose(a.grad, [0.6, 0], rtol=1e-5)


def test_cosine_lr():
    kw = dict(max_lr=1.0, total_steps=110, warmup_steps=10, min_lr=0.1)
    assert cosine_lr(0, **kw) == pytest.approx(0.1)
    assert cosine_lr(9, **kw) == pytest.approx(1.0)
    assert cosine_lr(10, **kw) == pytest.approx(1.0)
    assert cosine_lr(60, **kw) == pytest.approx(0.55)
    assert cosine_lr(110, **kw) == pytest.approx(0.1)
    lrs = [cosine_lr(s, **kw) for s in range(10, 111)]
    assert all(x >= y for x, y in zip(lrs, lrs[1:]))


def test_overfit_single_batch():
    """A small GPT must be able to memorize one batch."""
    data_rng = np.random.default_rng(1)
    idx = data_rng.integers(0, 10, (4, 16))
    targets = np.roll(idx, -1, axis=1)
    model = nn.GPT(vocab_size=10, block_size=16, n_layer=2, n_head=2, d_model=32, seed=0)
    opt = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(150):
        opt.zero_grad()
        _, loss = model(idx, targets)
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        opt.step()
    assert loss.item() < 0.05
