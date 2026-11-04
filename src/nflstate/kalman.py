"""Linear-Gaussian state-space model of NFL team strength.

State (dimension ``n_teams + 1``)::

    x = [r_1, ..., r_T, h]

``r_j`` is team *j*'s rating in points above an average team on a neutral
field, ``h`` is the league-wide home-field advantage (HFA).

Dynamics
    within a season, between consecutive weeks (``dt`` weeks apart)::

        r_j <- r_j + N(0, q_week * dt)            h <- h

    across an off-season::

        r_j <- rho * r_j + N(0, q_season)         h <- h + N(0, q_hfa)

    ``rho < 1`` is regression to the mean (draft, free agency, luck fading);
    ``q_hfa`` lets HFA drift from season to season.

Observation, one per game::

    margin = r_home - r_away + h * (1 - neutral) + d + N(0, sigma^2)

where ``d`` is a known offset (QB-change and rest adjustments, see
:class:`Params`). Because every piece is linear and Gaussian the Kalman filter
gives the *exact* posterior over ratings after each week, and the RTS smoother
gives the exact posterior given all games.

All games of a week are predicted from the same prior (built from strictly
earlier weeks) and then absorbed in one multivariate update.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import teams as team_list

LOG2PI = np.log(2 * np.pi)


# --------------------------------------------------------------------------
# generic Kalman primitives (used by the model and tested on toy systems)
# --------------------------------------------------------------------------
def predict(m: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Time update: ``x' = F x + w``, ``w ~ N(0, Q)``."""
    return F @ m, F @ P @ F.T + Q


def update(
    m: np.ndarray, P: np.ndarray, H: np.ndarray, R: np.ndarray, y: np.ndarray, d: np.ndarray | float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Measurement update for ``y = H x + d + v``, ``v ~ N(0, R)``.

    Returns
    -------
    m_post, P_post, y_hat, S, loglik
        posterior mean/covariance, predictive mean ``H m + d`` and covariance
        ``S = H P H' + R`` of ``y``, and the predictive log density of ``y``.
    """
    y_hat = H @ m + d
    S = H @ P @ H.T + R
    L = np.linalg.cholesky(S)
    resid = y - y_hat
    # K = P H' S^{-1} via two triangular solves
    PHt = P @ H.T
    K = np.linalg.solve(L.T, np.linalg.solve(L, PHt.T)).T
    m_post = m + K @ resid
    P_post = P - K @ PHt.T
    P_post = 0.5 * (P_post + P_post.T)  # guard against round-off asymmetry
    z = np.linalg.solve(L, resid)
    loglik = -0.5 * (len(y) * LOG2PI + 2 * np.log(np.diag(L)).sum() + z @ z)
    return m_post, P_post, y_hat, S, float(loglik)


def rts_smooth(
    m_f: list[np.ndarray], P_f: list[np.ndarray], m_p: list[np.ndarray], P_p: list[np.ndarray], F: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Rauch-Tung-Striebel smoother.

    ``m_p[t], P_p[t]`` is the prior at step ``t`` (after the time update with
    ``F[t]``), ``m_f[t], P_f[t]`` the posterior after step ``t``'s data.
    """
    n = len(m_f)
    m_s, P_s = [None] * n, [None] * n
    m_s[-1], P_s[-1] = m_f[-1], P_f[-1]
    for t in range(n - 2, -1, -1):
        J = np.linalg.solve(P_p[t + 1].T, (P_f[t] @ F[t + 1].T).T).T
        m_s[t] = m_f[t] + J @ (m_s[t + 1] - m_p[t + 1])
        P_s[t] = P_f[t] + J @ (P_s[t + 1] - P_p[t + 1]) @ J.T
    return m_s, P_s


# --------------------------------------------------------------------------
# NFL model
# --------------------------------------------------------------------------
@dataclass
class Params:
    """Hyperparameters of the team-strength model (points, points^2)."""

    sigma: float = 13.0        # game-level noise sd of the margin
    q_week: float = 0.6        # weekly rating innovation variance
    rho: float = 0.75          # off-season carry-over of ratings
    q_season: float = 6.0      # extra off-season variance
    q_hfa: float = 0.15        # season-to-season HFA innovation variance
    beta_qb: float = 0.0       # points added when a team starts a non-usual QB
    beta_rest: float = 0.0     # points per full week of rest advantage
    prior_sd: float = 6.0      # sd of ratings at the start of the first season
    hfa_mean0: float = 2.5     # prior mean of HFA in the first season
    hfa_sd0: float = 1.0       # prior sd of HFA in the first season

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FilterResult:
    """Output of :func:`run_filter`."""

    teams: list[str]
    games: pd.DataFrame                 # input games + pre-game predictions
    steps: pd.DataFrame                 # one row per (season, week) step
    loglik: np.ndarray                  # joint predictive log-lik per step
    m_f: list[np.ndarray] = field(repr=False, default_factory=list)
    P_f: list[np.ndarray] = field(repr=False, default_factory=list)
    m_p: list[np.ndarray] = field(repr=False, default_factory=list)
    P_p: list[np.ndarray] = field(repr=False, default_factory=list)
    F: list[np.ndarray] = field(repr=False, default_factory=list)

    def smooth(self) -> tuple[np.ndarray, np.ndarray]:
        """Smoothed means ``(n_steps, n_state)`` and sds ``(n_steps, n_state)``."""
        m_s, P_s = rts_smooth(self.m_f, self.P_f, self.m_p, self.P_p, self.F)
        return np.array(m_s), np.sqrt(np.array([np.diag(P) for P in P_s]))

    def filtered(self) -> tuple[np.ndarray, np.ndarray]:
        """Filtered (real-time) means and sds after each step."""
        return np.array(self.m_f), np.sqrt(np.array([np.diag(P) for P in self.P_f]))


def offsets(games: pd.DataFrame, p: Params) -> np.ndarray:
    """Known per-game shift of the home margin from QB changes and rest."""
    qb = games["home_qb_change"].to_numpy() - games["away_qb_change"].to_numpy()
    rest = (games["home_rest"] - games["away_rest"]).fillna(0).clip(-7, 7).to_numpy() / 7.0
    return p.beta_qb * qb + p.beta_rest * rest


def run_filter(games: pd.DataFrame, p: Params, store: bool = False) -> FilterResult:
    """Run the Kalman filter through ``games`` (sorted by season, week).

    Adds pre-game columns ``pred_margin``, ``pred_sd`` (predictive sd of the
    margin), ``p_home`` (P(home wins)) to a copy of ``games``. Each week's
    predictions use only games from earlier weeks.
    """
    tm = team_list(games)
    idx = {t: i for i, t in enumerate(tm)}
    T = len(tm)
    n = T + 1
    games = games.reset_index(drop=True)
    hi = games["home_team"].map(idx).to_numpy()
    ai = games["away_team"].map(idx).to_numpy()
    hfa_ind = 1.0 - games["neutral"].to_numpy()
    y_all = games["result"].to_numpy(dtype=float)
    d_all = offsets(games, p)

    m = np.zeros(n)
    m[T] = p.hfa_mean0
    P = np.diag(np.r_[np.full(T, p.prior_sd**2), p.hfa_sd0**2])

    F_season = np.diag(np.r_[np.full(T, p.rho), 1.0])
    Q_season = np.diag(np.r_[np.full(T, p.q_season), p.q_hfa])
    I = np.eye(n)
    q_week_diag = np.r_[np.full(T, p.q_week), 0.0]

    pred_mu = np.empty(len(games))
    pred_var = np.empty(len(games))
    lls, step_rows = [], []
    res = FilterResult(tm, games, pd.DataFrame(), np.array([]))

    keys = games[["season", "week"]].to_numpy()
    # boundaries of (season, week) blocks
    change = np.r_[True, (keys[1:] != keys[:-1]).any(axis=1)]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(games)]
    prev_season, prev_week = None, None
    for s, e in zip(starts, ends):
        season, week = keys[s]
        if prev_season is None:
            F, Q = I, np.zeros((n, n))
        elif season != prev_season:
            F, Q = F_season, Q_season
        else:
            F, Q = I, np.diag(q_week_diag * (week - prev_week))
        m, P = predict(m, P, F, Q)
        if store:
            res.m_p.append(m.copy()); res.P_p.append(P.copy()); res.F.append(F)

        k = e - s
        H = np.zeros((k, n))
        rows = np.arange(k)
        H[rows, hi[s:e]] = 1.0
        H[rows, ai[s:e]] = -1.0
        H[:, T] = hfa_ind[s:e]
        R = np.eye(k) * p.sigma**2
        m, P, y_hat, S, ll = update(m, P, H, R, y_all[s:e], d_all[s:e])
        pred_mu[s:e] = y_hat
        pred_var[s:e] = np.diag(S)
        lls.append(ll)
        step_rows.append((season, week, k))
        if store:
            res.m_f.append(m.copy()); res.P_f.append(P.copy())
        prev_season, prev_week = season, week

    out = games.copy()
    out["pred_margin"] = pred_mu
    out["pred_sd"] = np.sqrt(pred_var)
    out["p_home"] = norm.cdf(pred_mu / out["pred_sd"])
    res.games = out
    res.steps = pd.DataFrame(step_rows, columns=["season", "week", "n_games"])
    res.loglik = np.array(lls)
    return res


def cover_probability(pred_margin: np.ndarray, pred_sd: np.ndarray, spread_line: np.ndarray) -> np.ndarray:
    """P(home margin > spread_line) under the Normal predictive distribution."""
    return norm.cdf((pred_margin - spread_line) / pred_sd)
