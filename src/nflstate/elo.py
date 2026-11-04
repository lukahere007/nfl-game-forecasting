"""538-style Elo baseline (Silver's NFL Elo, without the QB module).

* expected score ``E = 1 / (1 + 10^(-(elo_h - elo_a + hfa) / 400))``
* update ``K * mov_mult * (S - E)`` with the margin-of-victory multiplier
  ``ln(|mov| + 1) * 2.2 / (0.001 * elo_diff_winner + 2.2)`` (dampens
  autocorrelation from favourites running up scores)
* between seasons ratings revert a fraction ``revert`` toward 1505
* point spread ~ Elo difference / 25
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EloParams:
    k: float = 20.0
    hfa: float = 55.0
    revert: float = 1 / 3
    mean: float = 1505.0


def expected(diff: np.ndarray | float) -> np.ndarray | float:
    """Elo win probability for rating difference ``diff`` (incl. HFA)."""
    return 1.0 / (1.0 + 10 ** (-np.asarray(diff) / 400.0))


def mov_multiplier(mov: float, elo_diff_winner: float) -> float:
    """538's margin-of-victory multiplier."""
    return np.log(abs(mov) + 1.0) * 2.2 / (0.001 * elo_diff_winner + 2.2)


def run_elo(games: pd.DataFrame, p: EloParams = EloParams()) -> pd.DataFrame:
    """Return ``games`` with pre-game ``elo_p_home`` and ``elo_spread`` columns.

    Ratings are updated game by game in chronological order; a game's
    prediction uses only ratings from before it is played.
    """
    games = games.reset_index(drop=True)
    rating: dict[str, float] = {}
    season_prev = None
    p_home = np.empty(len(games))
    spread = np.empty(len(games))
    for i, g in enumerate(games.itertuples(index=False)):
        if season_prev is not None and g.season != season_prev:
            rating = {t: r + p.revert * (p.mean - r) for t, r in rating.items()}
        season_prev = g.season
        rh = rating.get(g.home_team, p.mean)
        ra = rating.get(g.away_team, p.mean)
        diff = rh - ra + (0.0 if g.neutral else p.hfa)
        e = expected(diff)
        p_home[i], spread[i] = e, diff / 25.0
        s = 1.0 if g.result > 0 else 0.0 if g.result < 0 else 0.5
        win_diff = diff if g.result > 0 else -diff if g.result < 0 else 0.0
        shift = p.k * mov_multiplier(g.result, win_diff) * (s - e)
        rating[g.home_team] = rh + shift
        rating[g.away_team] = ra - shift
    out = games.copy()
    out["elo_p_home"] = p_home
    out["elo_spread"] = spread
    return out


def tune_elo(games: pd.DataFrame, train_start: int, train_end: int,
             ks=(10, 15, 20, 25, 30), hfas=(30, 45, 55, 65)) -> EloParams:
    """Grid-search K and HFA by log loss on training seasons (ties excluded)."""
    g = games[games["season"] <= train_end]
    best, best_ll = None, np.inf
    for k in ks:
        for h in hfas:
            p = EloParams(k=k, hfa=h)
            out = run_elo(g, p)
            sc = out[(out["season"] >= train_start) & (out["result"] != 0)]
            y = (sc["result"] > 0).to_numpy()
            q = np.clip(sc["elo_p_home"].to_numpy(), 1e-9, 1 - 1e-9)
            ll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
            if ll < best_ll:
                best, best_ll = p, ll
    return best
