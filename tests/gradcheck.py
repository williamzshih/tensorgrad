"""Finite-difference gradient checking for tensorgrad ops."""
import numpy as np

from tensorgrad.tensor import Tensor


def numerical_grads(f, arrays, eps=1e-6):
    """Central differences of the scalar function f(*arrays) w.r.t. each array."""
    grads = []
    for x in arrays:
        g = np.zeros_like(x)
        it = np.nditer(x, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            orig = x[i]
            x[i] = orig + eps
            hi = f(*arrays)
            x[i] = orig - eps
            lo = f(*arrays)
            x[i] = orig
            g[i] = (hi - lo) / (2 * eps)
        grads.append(g)
    return grads


def gradcheck(fn, arrays, rtol=1e-5, atol=1e-7, seed=0):
    """Check autodiff gradients of fn(*Tensors) against finite differences.

    A non-scalar output is reduced to a scalar with fixed random weights,
    sum(out * w), so every output element contributes a distinct gradient.
    """
    arrays = [np.array(a, dtype=np.float64) for a in arrays]
    probe = fn(*[Tensor(a) for a in arrays])
    w = np.random.default_rng(seed).standard_normal(probe.shape)

    def scalar(*arrs):
        return float((fn(*[Tensor(a) for a in arrs]).data * w).sum())

    inputs = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = fn(*inputs)
    out.backward(w)
    expected = numerical_grads(scalar, arrays)
    for k, (t, num) in enumerate(zip(inputs, expected)):
        assert t.grad is not None, f"input {k} got no gradient"
        np.testing.assert_allclose(t.grad, num, rtol=rtol, atol=atol,
                                   err_msg=f"gradient mismatch for input {k}")


def gradcheck_module(module, loss_fn, rtol=1e-5, atol=1e-7, eps=1e-6):
    """Gradcheck every parameter of `module` for the scalar loss_fn()."""
    module.zero_grad()
    loss_fn().backward()
    for name, p in module.named_parameters():
        num = numerical_grads(lambda _: loss_fn().item(), [p.data], eps)[0]
        np.testing.assert_allclose(p.grad, num, rtol=rtol, atol=atol,
                                   err_msg=f"gradient mismatch for {name}")
