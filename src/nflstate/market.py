"""Market-implied probabilities from closing moneylines and spreads."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm


def american_to_prob(odds: np.ndarray | float) -> np.ndarray:
    """Implied probability of American odds, *including* the bookmaker's margin.

    ``-150 -> 150/250 = 0.6``, ``+130 -> 100/230 = 0.4348``.
    """
    o = np.asarray(odds, dtype=float)
    if np.any(np.abs(o) < 100):
        raise ValueError("American odds must satisfy |odds| >= 100")
    a = np.abs(o)
    return np.where(o < 0, a, 100.0) / (a + 100.0)


def remove_vig(p_home_raw: np.ndarray, p_away_raw: np.ndarray) -> np.ndarray:
    """Proportional (multiplicative) de-vigging: ``p_home / (p_home + p_away)``.

    The overround ``p_home + p_away - 1`` is the bookmaker's margin; dividing
    by the booksum spreads it over both sides in proportion to their price.
    """
    p_home_raw, p_away_raw = np.asarray(p_home_raw, float), np.asarray(p_away_raw, float)
    return p_home_raw / (p_home_raw + p_away_raw)


def overround(home_ml, away_ml) -> np.ndarray:
    """Bookmaker margin implied by a pair of moneylines."""
    return american_to_prob(home_ml) + american_to_prob(away_ml) - 1.0


def fit_spread_sd(spread: np.ndarray, home_win: np.ndarray) -> float:
    """Fit ``s`` in ``P(home win) = Phi(spread / s)`` by maximum likelihood."""
    spread, y = np.asarray(spread, float), np.asarray(home_win, float)

    def nll(s):
        q = np.clip(norm.cdf(spread / s), 1e-12, 1 - 1e-12)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    return float(minimize_scalar(nll, bounds=(5, 30), method="bounded").x)


def market_prob(games: pd.DataFrame, spread_sd: float) -> pd.DataFrame:
    """Add ``mkt_p_home`` (no-vig moneyline, or spread-implied if missing).

    ``mkt_source`` records which was used.
    """
    out = games.copy()
    has_ml = out["home_moneyline"].notna() & out["away_moneyline"].notna()
    p_ml = remove_vig(american_to_prob(out["home_moneyline"].fillna(-110)),
                      american_to_prob(out["away_moneyline"].fillna(-110)))
    p_sp = norm.cdf(out["spread_line"].to_numpy(float) / spread_sd)
    out["mkt_p_home"] = np.where(has_ml, p_ml, p_sp)
    out["mkt_source"] = np.where(has_ml, "moneyline", "spread")
    return out
