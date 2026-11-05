"""Tensor: an np.ndarray plus the bookkeeping reverse-mode autodiff needs.

Each Tensor produced by an op stores
  _parents  : the input Tensors it was computed from
  _backward : a function mapping the upstream gradient dL/d(out) to a tuple
              of gradients dL/d(parent), one per parent (None = no gradient)
backward() walks the graph in reverse topological order and *accumulates*
those gradients into each parent's .grad, so a tensor used several times
receives the sum of its contributions.

The ops themselves live in ops.py; the operator overloads below just
forward to them.
"""
import contextlib

import numpy as np

_grad_enabled = True


@contextlib.contextmanager
def no_grad():
    """Disable graph construction (for evaluation / sampling)."""
    global _grad_enabled
    prev, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = prev


def is_grad_enabled():
    return _grad_enabled


class Tensor:
    def __init__(self, data, requires_grad=False, dtype=None):
        data = np.asarray(data, dtype=dtype)
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float64)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = ()
        self._backward = None
        self._op = ""

    @classmethod
    def _make(cls, data, parents, backward, op):
        """Create an op's output Tensor and record it in the graph if needed."""
        out = cls(data)
        if _grad_enabled and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = parents
            out._backward = backward
            out._op = op
        return out

    # --- basic properties --------------------------------------------------
    shape = property(lambda self: self.data.shape)
    ndim = property(lambda self: self.data.ndim)
    dtype = property(lambda self: self.data.dtype)
    size = property(lambda self: self.data.size)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        rg = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data!r}{rg})"

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def detach(self):
        return Tensor(self.data)

    def zero_grad(self):
        self.grad = None

    # --- backprop ----------------------------------------------------------
    def _topo_order(self):
        """Nodes reachable from self, each after all of its parents."""
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:  # iterative post-order DFS: no recursion limit
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
            elif id(node) not in visited:
                visited.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p in node._parents if p.requires_grad)
        return topo

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("backward() on a tensor that does not require grad")
        if grad is None:
            if self.size != 1:
                raise RuntimeError("grad must be given for non-scalar outputs")
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=self.dtype).reshape(self.shape)

        # Gradients of intermediate nodes live in this dict only; leaves
        # (no _backward) accumulate into .grad across backward() calls.
        grads = {id(self): grad}
        for node in reversed(self._topo_order()):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._backward is None:  # leaf
                node.grad = g.copy() if node.grad is None else node.grad + g
                continue
            for parent, pg in zip(node._parents, node._backward(g)):
                if pg is None or not parent.requires_grad:
                    continue
                assert pg.shape == parent.shape, (node._op, pg.shape, parent.shape)
                key = id(parent)
                grads[key] = pg if key not in grads else grads[key] + pg

    # --- operator overloads (implemented in ops.py) --------------------------
    def __add__(self, o): return _ops().add(self, o)
    def __radd__(self, o): return _ops().add(o, self)
    def __sub__(self, o): return _ops().sub(self, o)
    def __rsub__(self, o): return _ops().sub(o, self)
    def __mul__(self, o): return _ops().mul(self, o)
    def __rmul__(self, o): return _ops().mul(o, self)
    def __truediv__(self, o): return _ops().div(self, o)
    def __rtruediv__(self, o): return _ops().div(o, self)
    def __neg__(self): return _ops().neg(self)
    def __pow__(self, o): return _ops().pow(self, o)
    def __rpow__(self, o): return _ops().pow(o, self)
    def __matmul__(self, o): return _ops().matmul(self, o)
    def __getitem__(self, idx): return _ops().getitem(self, idx)

    def sum(self, axis=None, keepdims=False): return _ops().sum(self, axis, keepdims)
    def mean(self, axis=None, keepdims=False): return _ops().mean(self, axis, keepdims)
    def max(self, axis=None, keepdims=False): return _ops().max(self, axis, keepdims)
    def exp(self): return _ops().exp(self)
    def log(self): return _ops().log(self)
    def tanh(self): return _ops().tanh(self)
    def relu(self): return _ops().relu(self)
    def gelu(self): return _ops().gelu(self)
    def reshape(self, *shape): return _ops().reshape(self, *shape)
    def transpose(self, *axes): return _ops().transpose(self, *axes)
    permute = transpose

    @property
    def T(self):
        return _ops().transpose(self)


def _ops():
    from . import ops  # deferred: ops.py imports this module
    return ops
