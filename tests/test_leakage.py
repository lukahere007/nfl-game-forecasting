"""Predictions for week w must depend only on games played before week w."""

import numpy as np

from nflstate.data import qb_change_flags
from nflstate.elo import run_elo
from nflstate.kalman import Params, run_filter


def _cutoff_mask(g, season, week):
    return (g["season"] > season) | ((g["season"] == season) & (g["week"] >= week))


def test_kalman_predictions_ignore_current_and_future_results(toy_games):
    p = Params(beta_qb=-2, beta_rest=1)
    base = run_filter(toy_games, p).games
    future = _cutoff_mask(toy_games, 2011, 3)
    scrambled = toy_games.copy()
    scrambled.loc[future, "result"] = np.random.default_rng(5).integers(-40, 40, future.sum())
    new = run_filter(scrambled, p).games
    # every game up to and including week 3 of 2011 is predicted identically
    np.testing.assert_allclose(base.loc[future.values & (toy_games["season"] == 2011) & (toy_games["week"] == 3),
                                        "pred_margin"],
                               new.loc[future.values & (toy_games["season"] == 2011) & (toy_games["week"] == 3),
                                       "pred_margin"])
    np.testing.assert_allclose(base.loc[~future.values, ["pred_margin", "pred_sd"]],
                               new.loc[~future.values, ["pred_margin", "pred_sd"]])
    # and the change does propagate to later weeks (the test is not vacuous)
    later = _cutoff_mask(toy_games, 2011, 4).values
    assert not np.allclose(base.loc[later, "pred_margin"], new.loc[later, "pred_margin"])


def test_elo_predictions_ignore_current_and_future_results(toy_games):
    base = run_elo(toy_games)
    future = _cutoff_mask(toy_games, 2011, 3)
    scrambled = toy_games.copy()
    scrambled.loc[future, "result"] = -scrambled.loc[future, "result"] + 7
    new = run_elo(scrambled)
    first_future = np.flatnonzero(future.values)[0]
    np.testing.assert_allclose(base["elo_p_home"].iloc[: first_future + 1], new["elo_p_home"].iloc[: first_future + 1])


def test_qb_flags_use_only_earlier_games(toy_games):
    g = toy_games.copy()
    h0, a0 = qb_change_flags(g)
    future = _cutoff_mask(g, 2011, 3).values
    g.loc[future, "home_qb_id"] = "someone_else"
    h1, a1 = qb_change_flags(g)
    np.testing.assert_array_equal(h0[~future], h1[~future])
    np.testing.assert_array_equal(a0[~future], a1[~future])
