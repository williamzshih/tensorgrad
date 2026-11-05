"""M4: module shapes, save/load, end-to-end GPT gradcheck, causality."""
import numpy as np
import pytest

from tensorgrad import nn
from tensorgrad.tensor import Tensor
from tests.gradcheck import gradcheck_module

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


def tiny_gpt(**kw):
    cfg = dict(vocab_size=5, block_size=4, n_layer=1, n_head=2, d_model=8,
               seed=0, dtype=np.float64)
    cfg.update(kw)
    model = nn.GPT(**cfg)
    # Non-trivial LayerNorm params so their gradients are actually exercised.
    rng = np.random.default_rng(1)
    for name, p in model.named_parameters():
        p.data = p.data + 0.3 * rng.standard_normal(p.shape)
    return model


def test_shapes():
    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((2, 3, 8)).astype(np.float32))
    assert nn.Linear(8, 5, rng=rng)(x).shape == (2, 3, 5)
    assert nn.Embedding(10, 8, rng=rng)(np.array([[1, 2], [3, 9]])).shape == (2, 2, 8)
    assert nn.LayerNorm(8)(x).shape == (2, 3, 8)
    assert nn.CausalSelfAttention(8, 2, rng=rng)(x).shape == (2, 3, 8)
    assert nn.MLP(8, rng=rng)(x).shape == (2, 3, 8)
    assert nn.Block(8, 2, rng=rng)(x).shape == (2, 3, 8)
    gpt = nn.GPT(vocab_size=11, block_size=6, n_layer=2, n_head=2, d_model=8)
    logits, loss = gpt(np.zeros((3, 6), dtype=int), np.zeros((3, 6), dtype=int))
    assert logits.shape == (3, 6, 11) and loss.shape == ()
    assert logits.dtype == np.float32 and loss.dtype == np.float32


def test_parameter_discovery_and_count():
    gpt = nn.GPT(vocab_size=5, block_size=4, n_layer=2, n_head=2, d_model=8)
    names = [n for n, _ in gpt.named_parameters()]
    assert len(names) == len(set(names))
    assert "blocks.1.attn.qkv.weight" in names and "lm_head.weight" in names
    d, V, T, L = 8, 5, 4, 2
    per_block = 2 * 2 * d + (d * 3 * d + 3 * d) + (d * d + d) + (d * 4 * d + 4 * d) + (4 * d * d + d)
    assert gpt.num_parameters() == V * d + T * d + L * per_block + 2 * d + d * V


def test_save_load_roundtrip(tmp_path):
    a = tiny_gpt(seed=0)
    b = tiny_gpt(seed=123)
    idx = np.array([[1, 2, 3, 4]])
    assert not np.allclose(a(idx).data, b(idx).data)
    a.save(tmp_path / "ckpt.npz", note=np.array("hi"))
    b.load(tmp_path / "ckpt.npz")
    np.testing.assert_array_equal(a(idx).data, b(idx).data)


def test_load_rejects_mismatch():
    a = tiny_gpt()
    state = a.state_dict()
    state.pop("lm_head.weight")
    with pytest.raises(KeyError):
        a.load_state_dict(state)


def test_zero_grad():
    gpt = tiny_gpt()
    gpt(np.array([[0, 1, 2, 3]]), np.array([[1, 2, 3, 4]]))[1].backward()
    assert all(p.grad is not None for p in gpt.parameters())
    gpt.zero_grad()
    assert all(p.grad is None for p in gpt.parameters())


def test_gpt_end_to_end_gradcheck():
    gpt = tiny_gpt()
    idx = np.array([[0, 1, 2, 3], [4, 4, 1, 0]])
    targets = np.array([[1, 2, 3, 4], [4, 1, 0, 2]])
    gradcheck_module(gpt, lambda: gpt(idx, targets)[1])


def test_causality():
    gpt = tiny_gpt(block_size=8, n_layer=2)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 5, (1, 8))
    base = gpt(idx).data
    for t in range(8):
        changed = idx.copy()
        changed[0, t] = (changed[0, t] + 1) % 5
        out = gpt(changed).data
        np.testing.assert_array_equal(out[:, :t], base[:, :t])  # past is untouched
        assert not np.allclose(out[:, t], base[:, t])  # the changed position does move


def test_generate_shape_and_range():
    gpt = tiny_gpt()
    out = gpt.generate(np.array([[0, 1]]), 6, rng=np.random.default_rng(0), top_k=3)
    assert out.shape == (1, 8) and out.min() >= 0 and out.max() < 5


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_gpt_matches_torch():
    """Re-implement the same forward functionally in torch with our weights."""
    gpt = tiny_gpt(n_layer=2)
    idx = np.array([[0, 1, 2, 3], [4, 4, 1, 0]])
    targets = np.array([[1, 2, 3, 4], [4, 1, 0, 2]])
    P = {k: torch.tensor(v, requires_grad=True) for k, v in gpt.state_dict().items()}
    B, T, C, H = 2, 4, 8, 2

    def ln(x, pre):
        return F.layer_norm(x, (C,), P[pre + ".weight"], P[pre + ".bias"], eps=1e-5)

    def lin(x, pre):
        y = x @ P[pre + ".weight"]
        return y + P[pre + ".bias"] if pre + ".bias" in P else y

    x = P["wte.weight"][torch.tensor(idx)] + P["wpe.weight"][torch.arange(T)]
    for i in range(2):
        b = f"blocks.{i}."
        q, k, v = lin(ln(x, b + "ln1"), b + "attn.qkv").split(C, dim=-1)
        q, k, v = (t.reshape(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + lin(y.transpose(1, 2).reshape(B, T, C), b + "attn.proj")
        h = F.gelu(lin(ln(x, b + "ln2"), b + "mlp.fc"), approximate="tanh")
        x = x + lin(h, b + "mlp.proj")
    tlogits = lin(ln(x, "ln_f"), "lm_head")
    tloss = F.cross_entropy(tlogits.reshape(-1, 5), torch.tensor(targets).reshape(-1))
    tloss.backward()

    logits, loss = gpt(idx, targets)
    loss.backward()
    np.testing.assert_allclose(logits.data, tlogits.detach().numpy(), rtol=1e-9, atol=1e-12)
    assert loss.item() == pytest.approx(tloss.item(), rel=1e-12)
    for name, p in gpt.named_parameters():
        np.testing.assert_allclose(p.grad, P[name].grad.numpy(), rtol=1e-8, atol=1e-12,
                                   err_msg=name)
