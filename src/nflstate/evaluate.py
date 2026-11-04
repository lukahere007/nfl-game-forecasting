"""Proper scoring rules, calibration, bootstrap intervals and ATS summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

BREAKEVEN_110 = 110 / 210  # win rate needed to break even at -110


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean negative log-likelihood of binary outcomes."""
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of probability forecasts."""
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Equal-width reliability bins: mean forecast, observed rate, count, 95% CI."""
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"y": y, "p": p, "b": b})
    t = df.groupby("b").agg(p_mean=("p", "mean"), y_rate=("y", "mean"), n=("y", "size")).reset_index()
    lo, hi = wilson(t["y_rate"] * t["n"], t["n"])
    t["lo"], t["hi"] = lo, hi
    return t


def wilson(k: np.ndarray | float, n: np.ndarray | float, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion ``k / n``."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    ph = k / n
    centre = (ph + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def block_bootstrap_diff(loss_a: np.ndarray, loss_b: np.ndarray, blocks: np.ndarray,
                         n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of ``loss_a - loss_b`` with a 95% CI resampling whole blocks (weeks).

    Games in the same week share rating information, so resampling weeks
    rather than games keeps the interval honest.
    """
    diff = np.asarray(loss_a) - np.asarray(loss_b)
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, diff)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ub), size=(n_boot, len(ub)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    return float(diff.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def pointwise_log_loss(y, p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ats_by_disagreement(df: pd.DataFrame, edges=(0, 1, 2, 3, 4, np.inf)) -> pd.DataFrame:
    """Against-the-spread record of the model's side, bucketed by |model - line|.

    The model "takes" home if ``pred_margin > spread_line``. Pushes are dropped.
    """
    d = df[df["result"] != df["spread_line"]].copy()
    edge = d["pred_margin"] - d["spread_line"]
    d["abs_edge"] = edge.abs()
    d["model_right"] = (np.sign(edge) == np.sign(d["result"] - d["spread_line"])).astype(int)
    d["bucket"] = pd.cut(d["abs_edge"], list(edges), right=False)
    t = d.groupby("bucket", observed=True).agg(n=("model_right", "size"), wins=("model_right", "sum")).reset_index()
    t["win_rate"] = t["wins"] / t["n"]
    t["lo"], t["hi"] = wilson(t["wins"], t["n"])
    t["bucket"] = t["bucket"].astype(str)
    return t


def normal_margin_prob(mu, sd, threshold=0.0):
    """P(margin > threshold) for margin ~ N(mu, sd)."""
    return norm.cdf((np.asarray(mu) - threshold) / np.asarray(sd))
