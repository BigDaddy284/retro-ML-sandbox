"""T-704: a small neural network in NumPy."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning import Network
from retro_ai_lab.learning.network import SCHEMA_VERSION, huber


def inputs(n: int, width: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, (n, width))


def test_shapes_seeds_and_validation() -> None:
    net = Network((4, 8, 3), seed=1)
    out = net.forward(inputs(5, 4))
    assert out.shape == (5, 3) and np.isfinite(out).all()
    assert net.sizes == (4, 8, 3) and net.steps == 0
    assert np.array_equal(Network((4, 8, 3), seed=1).forward(inputs(5, 4)), out)
    assert not np.array_equal(Network((4, 8, 3), seed=2).forward(inputs(5, 4)), out)
    assert net.forward(inputs(1, 4)).shape == (1, 3)
    for sizes in ((4,), (4, 0, 3), (0, 3)):
        with pytest.raises(ConfigurationError):
            Network(sizes)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        Network((4, 3), learning_rate=0)
    with pytest.raises(ConfigurationError, match="4"):
        net.forward(inputs(2, 5))
    with pytest.raises(ConfigurationError):
        net.loss(inputs(2, 4), np.zeros((2, 2)))


def test_gradients_match_finite_differences() -> None:
    net = Network((3, 5, 4, 2), seed=7)
    x = inputs(6, 3, seed=3)
    targets = np.random.default_rng(4).normal(size=(6, 2)) * 3  # some beyond the Huber knee
    mask = np.random.default_rng(5).integers(0, 2, (6, 2)).astype(float)
    mask[0] = 1.0
    _, grads = net.gradients(x, targets, mask)
    params = net.parameters
    assert len(grads) == len(params) == 6
    eps = 1e-6
    for p, g in zip(params, grads, strict=True):
        assert g.shape == p.shape
        for idx in np.ndindex(*p.shape):
            saved = p[idx]
            p[idx] = saved + eps
            plus = net.loss(x, targets, mask)
            p[idx] = saved - eps
            minus = net.loss(x, targets, mask)
            p[idx] = saved
            numeric = (plus - minus) / (2 * eps)
            assert abs(numeric - g[idx]) <= 1e-6 * max(1.0, abs(numeric), abs(g[idx]))


def test_adam_fits_a_function_and_the_mask_hides_outputs() -> None:
    net = Network((2, 16, 2), seed=11, learning_rate=0.01)
    x = inputs(64, 2, seed=12)
    targets = np.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], axis=1)
    first = net.loss(x, targets)
    for _ in range(500):
        last = net.train_step(x, targets)
    assert last < first / 100 and net.steps == 500
    # Only the first output is trained; the second must be left alone by the loss.
    masked = Network((2, 16, 2), seed=11, learning_rate=0.01)
    mask = np.zeros_like(targets)
    mask[:, 0] = 1.0
    before = masked.forward(x)[:, 1].copy()
    assert masked.loss(x, targets, mask) == pytest.approx(
        huber(masked.forward(x)[:, 0] - targets[:, 0]).mean()
    )
    for _ in range(200):
        masked.train_step(x, targets, mask)
    assert np.abs(masked.forward(x)[:, 0] - targets[:, 0]).mean() < 0.2
    # Output 2 shares hidden units so it drifts, but far less than output 1 moved.
    moved_1 = np.abs(masked.forward(x)[:, 0] - Network((2, 16, 2), seed=11).forward(x)[:, 0])
    moved_2 = np.abs(masked.forward(x)[:, 1] - before)
    assert moved_2.mean() < moved_1.mean()


def test_copy_and_load_from() -> None:
    net = Network((3, 6, 2), seed=5)
    twin = net.copy()
    x = inputs(4, 3)
    assert np.array_equal(twin.forward(x), net.forward(x)) and twin.steps == 0
    net.train_step(x, np.ones((4, 2)))
    assert not np.array_equal(twin.forward(x), net.forward(x))
    twin.load_from(net)
    assert np.array_equal(twin.forward(x), net.forward(x))
    assert twin.steps == 0  # a target network keeps its own (unused) optimiser
    with pytest.raises(ConfigurationError):
        twin.load_from(Network((3, 7, 2)))


def test_json_round_trip_and_bad_data() -> None:
    net = Network((3, 4, 2), seed=9, learning_rate=0.02)
    x = inputs(5, 3)
    targets = np.ones((5, 2))
    for _ in range(3):
        net.train_step(x, targets)
    data = net.to_dict()
    assert data["schema"] == SCHEMA_VERSION and data["steps"] == 3
    again = Network.from_dict(data)
    assert again.sizes == net.sizes and again.learning_rate == 0.02 and again.steps == 3
    assert np.array_equal(again.forward(x), net.forward(x))
    assert again.train_step(x, targets) == net.train_step(x, targets)
    assert np.array_equal(again.forward(x), net.forward(x))
    assert again.to_dict() == net.to_dict()
    for bad in (
        "nope",
        {**data, "sizes": [3, 4]},
        {**data, "weights": data["weights"][:1]},
        {**data, "schema": SCHEMA_VERSION + 1},
        {**data, "learning_rate": -1},
    ):
        with pytest.raises(ConfigurationError):
            Network.from_dict(bad)  # type: ignore[arg-type]


@settings(max_examples=40)
@given(
    out=arrays(np.float64, (3, 2), elements=st.floats(-5, 5)),
    delta=arrays(np.float64, (3, 2), elements=st.floats(-5, 5)),
)
def test_huber_loss_is_never_negative_and_zero_at_the_target(
    out: np.ndarray, delta: np.ndarray
) -> None:
    assert huber(out - out).sum() == 0.0
    loss = huber(delta)
    assert (loss >= 0).all() and (loss <= np.abs(delta)).all()
    assert np.allclose(loss[np.abs(delta) <= 1], 0.5 * delta[np.abs(delta) <= 1] ** 2)
