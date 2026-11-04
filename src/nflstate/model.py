"""Hyperparameter estimation by maximising the one-step-ahead predictive likelihood.

The Kalman filter's prediction-error decomposition gives the exact marginal
likelihood of all observed margins as the product of each week's predictive
density given earlier weeks. Maximising it over the hyperparameters is
type-II maximum likelihood (empirical Bayes) and is, by construction, a
*forecasting* criterion: every term scores a genuine pre-game forecast.

Only games up to ``train_end`` are passed to the filter, so the test seasons
cannot influence the hyperparameters.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .kalman import Params, run_filter

#: Parameters that are estimated, with their transform (``log`` keeps a
#: variance positive, ``logit`` keeps ``rho`` in (0, 1), ``id`` is unconstrained).
FREE = {
    "sigma": "log", "q_week": "log", "rho": "logit", "q_season": "log",
    "q_hfa": "log", "beta_qb": "id", "beta_rest": "id",
}


def _to_unconstrained(p: Params, names: list[str]) -> np.ndarray:
    out = []
    for k in names:
        v, tr = getattr(p, k), FREE[k]
        out.append(np.log(v) if tr == "log" else np.log(v / (1 - v)) if tr == "logit" else v)
    return np.array(out)


def _from_unconstrained(z: np.ndarray, base: Params, names: list[str]) -> Params:
    vals = {}
    for k, v in zip(names, z):
        tr = FREE[k]
        vals[k] = float(np.exp(v) if tr == "log" else 1 / (1 + np.exp(-v)) if tr == "logit" else v)
    return replace(base, **vals)


def predictive_loglik(games: pd.DataFrame, p: Params, score_from: int) -> float:
    """Sum of weekly predictive log densities for seasons ``>= score_from``.

    Earlier seasons in ``games`` are burn-in: they move the ratings but are
    not scored.
    """
    res = run_filter(games, p)
    return float(res.loglik[res.steps["season"].to_numpy() >= score_from].sum())


def fit_params(
    games: pd.DataFrame,
    train_start: int,
    train_end: int,
    base: Params | None = None,
    free: list[str] | None = None,
    verbose: bool = False,
) -> tuple[Params, dict]:
    """Estimate hyperparameters on seasons ``train_start..train_end``.

    Games before ``train_start`` are burn-in; games after ``train_end`` are
    dropped before the filter is run.
    """
    base = base or Params()
    names = list(free or FREE)
    g = games[games["season"] <= train_end]

    def nll(z):
        val = -predictive_loglik(g, _from_unconstrained(z, base, names), train_start)
        if verbose:
            print(f"  nll={val:.2f}")
        return val

    z0 = _to_unconstrained(base, names)
    opt = minimize(nll, z0, method="L-BFGS-B", options={"maxiter": 200})
    p = _from_unconstrained(opt.x, base, names)
    n_games = int((g["season"] >= train_start).sum())
    info = {"loglik": -float(opt.fun), "n_games": n_games, "converged": bool(opt.success),
            "n_evals": int(opt.nfev), "free": names}
    return p, info
