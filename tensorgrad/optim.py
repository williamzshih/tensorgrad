"""Optimizers, gradient clipping and a learning-rate schedule."""
import math

import numpy as np


class Optimizer:
    """`params` is a list of Tensors, or a list of dicts {"params": [...], **overrides}
    (e.g. to turn off weight decay for biases and LayerNorm gains)."""

    def __init__(self, params, **defaults):
        params = list(params)
        if params and not isinstance(params[0], dict):
            params = [{"params": params}]
        self.param_groups = [{**defaults, **g, "params": list(g["params"])} for g in params]

    @property
    def lr(self):
        return self.param_groups[0]["lr"]

    @lr.setter
    def lr(self, value):
        for g in self.param_groups:
            g["lr"] = value

    def parameters(self):
        return [p for g in self.param_groups for p in g["params"]]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.velocity = {}

    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad + g["weight_decay"] * p.data if g["weight_decay"] else p.grad
                if g["momentum"]:
                    v = self.velocity.get(id(p))
                    v = grad.copy() if v is None else g["momentum"] * v + grad
                    self.velocity[id(p)] = v
                    grad = v
                p.data -= g["lr"] * grad


class AdamW(Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter), as in torch.optim.AdamW."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.t = 0
        self.m, self.v = {}, {}

    def step(self):
        self.t += 1
        for g in self.param_groups:
            lr, (b1, b2), eps, wd = g["lr"], g["betas"], g["eps"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                if wd:  # decoupled: shrink weights directly, not via the gradient
                    p.data -= lr * wd * p.data
                m = self.m.get(id(p), np.zeros_like(p.data))
                v = self.v.get(id(p), np.zeros_like(p.data))
                m = b1 * m + (1 - b1) * p.grad
                v = b2 * v + (1 - b2) * p.grad * p.grad
                self.m[id(p)], self.v[id(p)] = m, v
                m_hat = m / (1 - b1**self.t)  # bias correction
                v_hat = v / (1 - b2**self.t)
                p.data -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(p.dtype)


def clip_grad_norm(params, max_norm):
    """Scale all gradients so their global L2 norm is at most max_norm.
    Returns the norm before clipping."""
    grads = [p.grad for p in params if p.grad is not None]
    total = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
    scale = max_norm / (total + 1e-6)
    if scale < 1:
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
    return total


def cosine_lr(step, max_lr, total_steps, warmup_steps=0, min_lr=0.0):
    """Linear warmup to max_lr, then cosine decay to min_lr at total_steps."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
