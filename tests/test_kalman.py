"""The Kalman filter / smoother against closed-form and brute-force answers."""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

from nflstate.kalman import Params, predict, rts_smooth, run_filter, update


def test_single_game_update_matches_gaussian_conditioning():
    # two teams + HFA; prior N(m, P); one home game for team 0
    m = np.array([1.0, -0.5, 2.5])
    P = np.diag([16.0, 9.0, 1.0])
    H = np.array([[1.0, -1.0, 1.0]])
    sigma2, y = 169.0, np.array([10.0])
    m_post, P_post, y_hat, S, ll = update(m, P, H, np.eye(1) * sigma2, y)

    # closed form: s = H P H' + sigma2, posterior mean = m + P H' (y - Hm) / s
    s = (H @ P @ H.T).item() + sigma2
    np.testing.assert_allclose(S, [[s]])
    np.testing.assert_allclose(y_hat, H @ m)
    np.testing.assert_allclose(m_post, m + (P @ H.T).ravel() * (y - H @ m) / s)
    np.testing.assert_allclose(P_post, P - np.outer(P @ H.T, H @ P) / s)
    np.testing.assert_allclose(ll, norm.logpdf(y[0], (H @ m)[0], np.sqrt(s)))


def test_single_game_update_matches_brute_force_grid():
    # posterior of team 0's rating by numerical integration on a grid
    m = np.array([0.0, 0.0])
    P = np.diag([25.0, 4.0])
    H = np.array([[1.0, -1.0]])
    y, sigma = 14.0, 12.0
    m_post, P_post, *_ = update(m, P, H, np.eye(1) * sigma**2, np.array([y]))
    g = np.linspace(-40, 40, 801)
    r1, r2 = np.meshgrid(g, g, indexing="ij")
    w = norm.pdf(r1, 0, 5) * norm.pdf(r2, 0, 2) * norm.pdf(y, r1 - r2, sigma)
    w /= w.sum()
    mean1 = (w * r1).sum()
    var1 = (w * r1**2).sum() - mean1**2
    np.testing.assert_allclose(m_post[0], mean1, atol=1e-3)
    np.testing.assert_allclose(P_post[0, 0], var1, atol=1e-2)


def test_batch_week_update_equals_sequential_updates():
    rng = np.random.default_rng(1)
    n = 5
    A = rng.normal(size=(n, n))
    m, P = rng.normal(size=n), A @ A.T + np.eye(n)
    H = np.array([[1, -1, 0, 0, 1], [0, 0, 1, -1, 1.0]])
    y = np.array([3.0, -7.0])
    R = np.eye(2) * 100
    mb, Pb, _, _, llb = update(m, P, H, R, y)
    ms, Ps, ll_tot = m, P, 0.0
    for i in range(2):
        ms, Ps, _, _, ll = update(ms, Ps, H[i:i + 1], R[i:i + 1, i:i + 1], y[i:i + 1])
        ll_tot += ll
    np.testing.assert_allclose(mb, ms)
    np.testing.assert_allclose(Pb, Ps)
    np.testing.assert_allclose(llb, ll_tot)


def _batch_posterior(m0, P0, Fs, Qs, Hs, R, ys):
    """Brute force: stack all states and observations into one Gaussian and condition."""
    n, T = len(m0), len(Fs)
    # x_t = F_t x_{t-1} + w_t with x_{-1} ~ N(m0, P0) and F_0 = I, Q_0 = 0
    mean = []
    cov = np.zeros((n * T, n * T))
    # build x as linear map of [x_init, w_0..w_{T-1}]
    blocks = []
    M = np.eye(n)
    lin = []  # lin[t]: list of matrices mapping each source to x_t
    for t in range(T):
        M = Fs[t] @ M
        prev = [Fs[t] @ L for L in lin[-1]] if lin else []
        lin.append([M] + prev[1:] + [np.eye(n)] if lin else [M, np.eye(n)])
    src_cov = [P0] + list(Qs)
    for t in range(T):
        coeffs = lin[t] + [np.zeros((n, n))] * (T + 1 - len(lin[t]))
        blocks.append(coeffs)
        mean.append(coeffs[0] @ m0)
    for a in range(T):
        for b in range(T):
            cov[a * n:(a + 1) * n, b * n:(b + 1) * n] = sum(
                blocks[a][k] @ src_cov[k] @ blocks[b][k].T for k in range(T + 1))
    mean = np.concatenate(mean)
    Hbig = np.zeros((sum(len(y) for y in ys), n * T))
    r = 0
    for t, H in enumerate(Hs):
        Hbig[r:r + len(H), t * n:(t + 1) * n] = H
        r += len(H)
    y = np.concatenate(ys)
    S = Hbig @ cov @ Hbig.T + np.eye(len(y)) * R
    K = cov @ Hbig.T @ np.linalg.inv(S)
    post_m = mean + K @ (y - Hbig @ mean)
    post_P = cov - K @ Hbig @ cov
    loglik = multivariate_normal(Hbig @ mean, S).logpdf(y)
    return post_m.reshape(T, n), np.array([post_P[t * n:(t + 1) * n, t * n:(t + 1) * n] for t in range(T)]), loglik


def test_smoother_and_likelihood_match_brute_force_toy_league():
    # 2 teams + HFA, 3 steps: week 1, week 2, then a new season
    n = 3
    m0, P0 = np.array([0.0, 0.0, 2.5]), np.diag([36.0, 36.0, 1.0])
    rho, qw, qs, qh = 0.7, 0.5, 8.0, 0.2
    Fs = [np.eye(n), np.eye(n), np.diag([rho, rho, 1.0])]
    Qs = [np.zeros((n, n)), np.diag([qw, qw, 0]), np.diag([qs, qs, qh])]
    Hs = [np.array([[1.0, -1.0, 1.0]]), np.array([[-1.0, 1.0, 1.0]]), np.array([[1.0, -1.0, 0.0]])]
    ys = [np.array([10.0]), np.array([3.0]), np.array([-6.0])]
    R = 169.0

    m, P = m0, P0
    m_f, P_f, m_p, P_p, ll = [], [], [], [], 0.0
    for F, Q, H, y in zip(Fs, Qs, Hs, ys):
        m, P = predict(m, P, F, Q)
        m_p.append(m); P_p.append(P)
        m, P, _, _, l = update(m, P, H, np.eye(1) * R, y)
        m_f.append(m); P_f.append(P); ll += l
    m_s, P_s = rts_smooth(m_f, P_f, m_p, P_p, Fs)

    bm, bP, bll = _batch_posterior(m0, P0, Fs, Qs, Hs, R, ys)
    np.testing.assert_allclose(np.array(m_s), bm, atol=1e-8)
    np.testing.assert_allclose(np.array(P_s), bP, atol=1e-8)
    np.testing.assert_allclose(ll, bll, atol=1e-8)


def test_run_filter_matches_primitives_on_two_team_league():
    games = pd.DataFrame(dict(
        game_id=["g1", "g2", "g3"], season=[2010, 2010, 2011], week=[1, 3, 1],
        home_team=["A", "B", "A"], away_team=["B", "A", "B"], result=[10.0, 3.0, -6.0],
        neutral=[0, 0, 1], home_rest=[7, 7, 7], away_rest=[7, 7, 7],
        home_qb_change=[0, 0, 0], away_qb_change=[0, 0, 0]))
    p = Params(sigma=13, q_week=0.5, rho=0.7, q_season=8, q_hfa=0.2, prior_sd=6, hfa_mean0=2.5, hfa_sd0=1)
    res = run_filter(games, p, store=True)
    m, P = np.array([0, 0, 2.5]), np.diag([36.0, 36.0, 1.0])
    Fs = [np.eye(3), np.eye(3), np.diag([0.7, 0.7, 1])]
    Qs = [np.zeros((3, 3)), np.diag([1.0, 1.0, 0]), np.diag([8, 8, 0.2])]  # week gap of 2
    Hs = [[1, -1, 1], [-1, 1, 1], [1, -1, 0]]
    for t in range(3):
        m, P = predict(m, P, Fs[t], Qs[t])
        H = np.array([Hs[t]], float)
        mu, s = (H @ m).item(), np.sqrt(H @ P @ H.T + 169).item()
        assert np.isclose(res.games.loc[t, "pred_margin"], mu)
        assert np.isclose(res.games.loc[t, "pred_sd"], s)
        assert np.isclose(res.games.loc[t, "p_home"], norm.cdf(mu / s))
        m, P, *_ = update(m, P, H, np.eye(1) * 169, np.array([games.result[t]]))
    np.testing.assert_allclose(res.m_f[-1], m)


def test_offsets_shift_prediction():
    games = pd.DataFrame(dict(
        game_id=["g1"], season=[2010], week=[1], home_team=["A"], away_team=["B"], result=[3.0],
        neutral=[0], home_rest=[14], away_rest=[7], home_qb_change=[1], away_qb_change=[0]))
    base = run_filter(games, Params()).games.loc[0, "pred_margin"]
    adj = run_filter(games, Params(beta_qb=-3.0, beta_rest=1.5)).games.loc[0, "pred_margin"]
    assert np.isclose(adj - base, -3.0 + 1.5)
