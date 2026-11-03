"""A small neural network in NumPy (DESIGN.md §6.1; ADR-009).

Dense layers with ReLU between them and a linear output, a Huber loss on
the outputs a mask selects (a DQN only learns about the move it made), Adam,
copies for a target network and a JSON round trip. Parameters are float64:
the networks here are tiny and exactness is worth more than speed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from retro_ai_lab.core import ConfigurationError

SCHEMA_VERSION = 1
DEFAULT_LEARNING_RATE = 1e-3
HUBER_DELTA = 1.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

Array = NDArray[np.float64]


def huber(delta: Array, knee: float = HUBER_DELTA) -> Array:
    """Element-wise Huber loss: quadratic near zero, linear beyond ``knee``."""
    a = np.abs(delta)
    return np.where(a <= knee, 0.5 * delta * delta, knee * (a - 0.5 * knee))


def _huber_grad(delta: Array, knee: float = HUBER_DELTA) -> Array:
    return np.clip(delta, -knee, knee)


class Network:
    def __init__(
        self,
        sizes: Sequence[int],
        *,
        seed: int | None = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> None:
        sizes = tuple(sizes)
        if len(sizes) < 2 or any(not isinstance(n, int) or n < 1 for n in sizes):
            raise ConfigurationError("A network needs at least two layer sizes, all >= 1.")
        if not (isinstance(learning_rate, int | float) and 0 < learning_rate < 1):
            raise ConfigurationError("learning_rate must be between 0 and 1.")
        self._sizes = sizes
        self._lr = float(learning_rate)
        rng = np.random.default_rng(seed)
        self._weights: list[Array] = []
        self._biases: list[Array] = []
        for fan_in, fan_out in pairwise(sizes):
            scale = math.sqrt(2.0 / fan_in)  # He initialisation for ReLU
            self._weights.append(rng.normal(0.0, scale, (fan_in, fan_out)))
            self._biases.append(np.zeros(fan_out))
        self._steps = 0
        self._m = [np.zeros_like(p) for p in self.parameters]
        self._v = [np.zeros_like(p) for p in self.parameters]

    # ------------------------------------------------------------ queries

    @property
    def sizes(self) -> tuple[int, ...]:
        return self._sizes

    @property
    def learning_rate(self) -> float:
        return self._lr

    @property
    def steps(self) -> int:
        """Adam updates applied so far."""
        return self._steps

    @property
    def parameters(self) -> list[Array]:
        """The live arrays, weight then bias per layer (edit in place for checks)."""
        out: list[Array] = []
        for w, b in zip(self._weights, self._biases, strict=True):
            out.extend((w, b))
        return out

    # ------------------------------------------------------------ forward / backward

    def _check(self, x: Any, targets: Any = None, mask: Any = None) -> tuple[Array, Array, Array]:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self._sizes[0]:
            raise ConfigurationError(
                f"Input must be (n, {self._sizes[0]}), got shape {tuple(x.shape)}."
            )
        if targets is None:
            return x, np.empty((0, 0)), np.empty((0, 0))
        t = np.asarray(targets, dtype=np.float64)
        shape = (x.shape[0], self._sizes[-1])
        if t.shape != shape:
            raise ConfigurationError(f"Targets must be {shape}, got shape {tuple(t.shape)}.")
        m = np.ones(shape) if mask is None else np.asarray(mask, dtype=np.float64)
        if m.shape != shape:
            raise ConfigurationError(f"Mask must be {shape}, got shape {tuple(m.shape)}.")
        return x, t, m

    def _run(self, x: Array) -> tuple[list[Array], list[Array]]:
        """Activations after each layer and the pre-activations of the hidden ones."""
        acts = [x]
        pres: list[Array] = []
        last = len(self._weights) - 1
        for i, (w, b) in enumerate(zip(self._weights, self._biases, strict=True)):
            z = acts[-1] @ w + b
            if i < last:
                pres.append(z)
                z = np.maximum(z, 0.0)
            acts.append(z)
        return acts, pres

    def forward(self, x: Any) -> Array:
        x, _, _ = self._check(x)
        return self._run(x)[0][-1]

    def loss(self, x: Any, targets: Any, mask: Any = None) -> float:
        x, t, m = self._check(x, targets, mask)
        delta = (self._run(x)[0][-1] - t) * m
        return float(huber(delta).sum() / x.shape[0])

    def gradients(self, x: Any, targets: Any, mask: Any = None) -> tuple[float, list[Array]]:
        """The loss and its gradient for every parameter, in :attr:`parameters` order."""
        x, t, m = self._check(x, targets, mask)
        acts, pres = self._run(x)
        n = x.shape[0]
        delta = (acts[-1] - t) * m
        loss = float(huber(delta).sum() / n)
        grad = _huber_grad(delta) * m / n  # d loss / d output
        grads: list[Array] = []
        for i in range(len(self._weights) - 1, -1, -1):
            grads[:0] = [acts[i].T @ grad, grad.sum(axis=0)]
            if i > 0:
                grad = (grad @ self._weights[i].T) * (pres[i - 1] > 0)
        return loss, grads

    def train_step(self, x: Any, targets: Any, mask: Any = None) -> float:
        loss, grads = self.gradients(x, targets, mask)
        self._steps += 1
        t = self._steps
        lr_t = self._lr * math.sqrt(1 - ADAM_BETA2**t) / (1 - ADAM_BETA1**t)
        for p, g, m, v in zip(self.parameters, grads, self._m, self._v, strict=True):
            m *= ADAM_BETA1
            m += (1 - ADAM_BETA1) * g
            v *= ADAM_BETA2
            v += (1 - ADAM_BETA2) * g * g
            p -= lr_t * m / (np.sqrt(v) + ADAM_EPS)
        return loss

    # ------------------------------------------------------------ copies

    def copy(self) -> Network:
        """Same weights, fresh optimiser: a target network."""
        twin = Network(self._sizes, learning_rate=self._lr)
        twin.load_from(self)
        return twin

    def load_from(self, other: Network) -> None:
        """Take ``other``'s weights (the optimiser state and step count stay)."""
        if other.sizes != self._sizes:
            raise ConfigurationError(
                f"Can't load a {other.sizes} network into a {self._sizes} one."
            )
        for mine, theirs in zip(self.parameters, other.parameters, strict=True):
            mine[...] = theirs

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "sizes": list(self._sizes),
            "learning_rate": self._lr,
            "steps": self._steps,
            "weights": [w.tolist() for w in self._weights],
            "biases": [b.tolist() for b in self._biases],
            "adam_m": [m.tolist() for m in self._m],
            "adam_v": [v.tolist() for v in self._v],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Network:
        if not isinstance(data, dict):
            raise ConfigurationError("Bad network data: not a mapping.")
        try:
            schema = int(data.get("schema", SCHEMA_VERSION))
            if schema > SCHEMA_VERSION:
                raise ConfigurationError(f"This network is from a newer version (schema {schema}).")
            net = cls(
                tuple(int(n) for n in data["sizes"]),
                learning_rate=float(data.get("learning_rate", DEFAULT_LEARNING_RATE)),
            )
            steps = int(data.get("steps", 0))
            if steps < 0:
                raise ValueError("steps must be >= 0")
            net._steps = steps
            for store, key in (
                (net._weights, "weights"),
                (net._biases, "biases"),
                (net._m, "adam_m"),
                (net._v, "adam_v"),
            ):
                items = data[key]
                if len(items) != len(store):
                    raise ValueError(f"{key} holds {len(items)} arrays, not {len(store)}")
                for target, item in zip(store, items, strict=True):
                    arr = np.asarray(item, dtype=np.float64)
                    if arr.shape != target.shape:
                        raise ValueError(f"{key} shape {arr.shape} should be {target.shape}")
                    target[...] = arr
            return net
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad network data: {exc}") from exc


__all__ = ["DEFAULT_LEARNING_RATE", "HUBER_DELTA", "SCHEMA_VERSION", "Network", "huber"]
