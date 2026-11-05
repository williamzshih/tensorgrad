import math
import random

import pytest

from tensorgrad.value import Value


def numgrad(f, xs, i, eps=1e-6):
    hi = list(xs); hi[i] += eps
    lo = list(xs); lo[i] -= eps
    return (f(*hi) - f(*lo)) / (2 * eps)


def check(fn, xs):
    vs = [Value(x) for x in xs]
    out = fn(*vs)
    out.backward()
    f = lambda *a: fn(*[Value(x) for x in a]).data
    for i, v in enumerate(vs):
        assert v.grad == pytest.approx(numgrad(f, xs, i), rel=1e-5, abs=1e-8)


CASES = {
    "add": (lambda a, b: a + b, [1.3, -0.7]),
    "mul": (lambda a, b: a * b, [1.3, -0.7]),
    "sub_div": (lambda a, b: (a - b) / (a * b), [1.3, -0.7]),
    "pow": (lambda a: a**3 + a**-2 + a**0.5, [1.7]),
    "exp_log": (lambda a: (a * 2).exp().log() + a.log(), [0.9]),
    "tanh": (lambda a: a.tanh(), [0.4]),
    "relu": (lambda a, b: a.relu() * b.relu(), [0.4, -0.3]),
    "reuse": (lambda a: a * a + a * 3 + a.exp() * a, [0.8]),
    "rops": (lambda a: 2 - a + 3 / a + 1 + 2 * a, [1.1]),
}


@pytest.mark.parametrize("name", CASES)
def test_gradcheck(name):
    fn, xs = CASES[name]
    check(fn, xs)


def test_deep_graph_no_recursion_error():
    x = Value(1.0)
    y = x
    for _ in range(5000):
        y = y + 0.0
    y.backward()
    assert x.grad == 1.0


def make_moons(n, rng):
    data = []
    for i in range(n):
        t = rng.uniform(0, math.pi)
        if i % 2 == 0:
            x, y, label = math.cos(t), math.sin(t), -1
        else:
            x, y, label = 1 - math.cos(t), 0.5 - math.sin(t), 1
        data.append((x + rng.gauss(0, 0.1), y + rng.gauss(0, 0.1), label))
    return data


def test_mlp_trains_on_moons():
    rng = random.Random(0)
    data = make_moons(100, rng)
    H = 16
    W1 = [[Value(rng.gauss(0, 1)) for _ in range(2)] for _ in range(H)]
    b1 = [Value(0.0) for _ in range(H)]
    W2 = [Value(rng.gauss(0, 1) / math.sqrt(H)) for _ in range(H)]
    b2 = Value(0.0)
    params = [w for row in W1 for w in row] + b1 + W2 + [b2]

    def forward(x, y):
        h = [(W1[j][0] * x + W1[j][1] * y + b1[j]).tanh() for j in range(H)]
        return sum((W2[j] * h[j] for j in range(H)), b2)

    for step in range(100):
        # SVM hinge loss + L2, as in the classic micrograd demo
        scores = [forward(x, y) for x, y, _ in data]
        losses = [(1 - lab * s).relu() for (_, _, lab), s in zip(data, scores)]
        loss = sum(losses) * (1.0 / len(losses)) + 1e-4 * sum(p * p for p in params)
        for p in params:
            p.grad = 0.0
        loss.backward()
        lr = 1.0 - 0.9 * step / 100
        for p in params:
            p.data -= lr * p.grad

    acc = sum((forward(x, y).data > 0) == (lab > 0) for x, y, lab in data) / len(data)
    assert acc > 0.95
