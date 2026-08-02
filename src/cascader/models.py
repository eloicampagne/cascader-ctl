"""Local models: the per-task hypothesis a cascade transports.

A local model is any ``torch.nn.Module`` mapping ``(n, d)`` features to ``(n,)``
outputs. CTL transfers a parameter vector along the cascade, so every task must
share one parameter space; heterogeneous label spaces or per-task heads fall
outside the method as published.

``linear`` matches the contraction analysis of Section 2.3 exactly. ``mlp`` is
the non-convex model of the appendix: one hidden layer of 32 units with ReLU,
small enough that the comparison stays about the geometry rather than about
capacity.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn

__all__ = ["LinearHead", "MLPHead", "make_model", "MODELS", "model_factory"]


class LinearHead(nn.Module):
    """Affine map to a scalar. The regime the theory covers."""

    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).reshape(-1)


class MLPHead(nn.Module):
    """One hidden ReLU layer, then a scalar output."""

    def __init__(self, d: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


MODELS: Dict[str, Callable[..., nn.Module]] = {
    "linear": LinearHead,
    "mlp": MLPHead,
}


def make_model(kind: str, d: int, **kwargs: object) -> nn.Module:
    """Build a local model by name. ``kind`` is ``'linear'`` or ``'mlp'``."""
    try:
        cls = MODELS[kind]
    except KeyError:
        raise KeyError(f"unknown model {kind!r}; choose from {sorted(MODELS)}") from None
    return cls(d, **kwargs)  # type: ignore[arg-type]


def model_factory(kind: str, d: int, **kwargs: object) -> Callable[[], nn.Module]:
    """Return a zero-argument callable producing a fresh model each call.

    The cascade needs a factory rather than an instance: the root is built from
    scratch, and a fresh initialization is also what the Individual baseline
    gives every task.
    """
    return lambda: make_model(kind, d, **kwargs)
