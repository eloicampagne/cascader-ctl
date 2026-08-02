"""Per-collection hyperparameters, as published.

These are the values the paper's tables were produced with. The learning rates
under ``eta`` come from an Optuna search whose objective is a CTL-MSTc cascade's
validation loss, and are then applied to every method -- which shares the tuning
pipeline but not the rate. That is not a neutral choice: on WEAVE-UK the
cascades and Individual prefer rates an order of magnitude apart, so the shared
value handicaps the baseline. The per-method retuning that answers this is
:mod:`cascader.experiments.permethod_eta`, and the cascade still leads.

``mlp_eta`` is recalibrated per collection on the Individual baseline, so the
non-convex comparison is internal to one protocol rather than inheriting a rate
tuned for a linear model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

__all__ = ["DatasetConfig", "CONFIGS", "get_config"]


@dataclass(frozen=True)
class DatasetConfig:
    """Published hyperparameters for one collection.

    Attributes
    ----------
    eta
        Learning rate for the linear local model.
    mlp_eta
        Learning rate for the one-hidden-layer model, recalibrated on Individual.
    reg
        L2 coefficient added to the gradient. Zero on the regression collections.
    b_proto_frac
        Share of the budget reserved for the root under the uniform allocation.
    mstc_lambda
        Coverage weight of the CTL-MSTc construction. A wide interior plateau:
        both endpoints degrade and the optimum is interior.
    root_frac
        Share reserved for the root under the SLE allocation.
    task
        ``'regression'`` or ``'classification'``.
    T
        Number of tasks in the published collection.
    """

    eta: float
    mlp_eta: float
    reg: float
    b_proto_frac: float
    mstc_lambda: float
    task: str
    T: int
    root_frac: float = 0.10


CONFIGS: Dict[str, DatasetConfig] = {
    "Syn-10": DatasetConfig(
        eta=0.0012710047170539321, mlp_eta=0.2, reg=0.0,
        b_proto_frac=0.14414738903213942, mstc_lambda=0.6720291677813087,
        task="regression", T=200),
    "WEAVE-UK": DatasetConfig(
        eta=0.0015881946767357276, mlp_eta=0.2, reg=0.0,
        b_proto_frac=0.1526099724429147, mstc_lambda=0.6,
        task="regression", T=500),
    "FMNIST": DatasetConfig(
        eta=0.2955513127628392, mlp_eta=0.2, reg=0.00790085761024867,
        b_proto_frac=0.10, mstc_lambda=0.005891330018108831,
        task="classification", T=200),
    "CIFAR": DatasetConfig(
        eta=0.0670885289345722, mlp_eta=0.1, reg=0.00044369124423313027,
        b_proto_frac=0.10113798086175203, mstc_lambda=0.2588665932490024,
        task="classification", T=200),
    # CelebA: frozen ResNet-18 features with a trained head. The published study
    # tuned lr and reg by Optuna without storing the literals, so eta here is
    # calibrated on Individual like mlp_eta; mstc_lambda and the SLE allocation
    # are the notebook's.
    "CelebA": DatasetConfig(
        eta=0.2, mlp_eta=0.05, reg=1e-3,
        b_proto_frac=0.10, mstc_lambda=0.196,
        task="classification", T=400),
}


def get_config(name: str) -> DatasetConfig:
    """Look up a published configuration by collection name."""
    try:
        return CONFIGS[name]
    except KeyError:
        raise KeyError(
            f"unknown collection {name!r}; choose from {sorted(CONFIGS)}"
        ) from None
