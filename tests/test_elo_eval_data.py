import numpy as np
import pandas as pd
import pytest

from nflstate.data import FRANCHISE, clean_games, qb_change_flags
from nflstate.elo import EloParams, expected, mov_multiplier, run_elo
from nflstate.evaluate import ats_by_disagreement, block_bootstrap_diff, brier, log_loss, wilson


# ---------------- Elo ----------------
def test_elo_expected_and_multiplier():
    assert expected(0) == pytest.approx(0.5)
    assert expected(400) == pytest.approx(10 / 11)
    assert mov_multiplier(0, 0) == 0
    # a favourite's blowout counts for less than an underdog's
    assert mov_multiplier(14, 200) < mov_multiplier(14, -200)


def test_elo_first_games_use_initial_ratings(toy_games):
    p = EloParams()
    out = run_elo(toy_games, p)
    # the first two games of week 1 involve four distinct, unrated teams
    first = out.iloc[:2]
    expect = expected(np.where(first["neutral"] == 1, 0.0, p.hfa))
    np.testing.assert_allclose(first["elo_p_home"], expect)
    assert out["elo_p_home"].between(0, 1).all()


# ---------------- scoring ----------------
def test_scores():
    y = np.array([1, 0, 1, 0])
    assert log_loss(y, np.full(4, 0.5)) == pytest.approx(np.log(2))
    assert brier(y, np.full(4, 0.5)) == pytest.approx(0.25)
    assert brier(y, y) == 0


def test_wilson_interval():
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)


def test_block_bootstrap_constant_difference():
    a, b = np.full(10, 0.7), np.full(10, 0.6)
    m, lo, hi = block_bootstrap_diff(a, b, np.repeat(np.arange(5), 2), n_boot=100)
    assert m == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


def test_ats_by_disagreement_counts_sides_and_drops_pushes():
    df = pd.DataFrame(dict(pred_margin=[5.0, -2.0, 1.0, 4.0], spread_line=[3.0, 1.0, 1.5, 3.0],
                           result=[7.0, 3.0, 0.0, 3.0]))
    t = ats_by_disagreement(df, edges=(0, np.inf))
    # game 1: model home, home covers -> right; game 2: model away, home covers -> wrong;
    # game 3: model away, away covers -> right; game 4: push -> dropped
    assert t["n"].iloc[0] == 3 and t["wins"].iloc[0] == 2


# ---------------- data ----------------
def test_qb_change_flags_toy():
    g = pd.DataFrame(dict(
        season=[2020] * 4 + [2021], home_team=["A", "B", "A", "A", "A"], away_team=["B", "A", "B", "B", "B"],
        home_qb_id=["a1", "b1", "a2", "a1", "a9"], away_qb_id=["b1", "a1", "b1", "b2", "b1"]))
    h, a = qb_change_flags(g)
    # game 0: first game, never flagged; game 1: A's a1 is usual, B's b1 usual
    # game 2: A starts a2 (usual a1) -> flagged; game 3: A back to a1, B starts b2 -> flagged
    # game 4: new season, first game -> not flagged
    np.testing.assert_array_equal(h, [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(a, [0, 0, 0, 1, 0])


def test_clean_games_maps_franchises_and_sorts():
    raw = pd.DataFrame(dict(
        game_id=["x2", "x1"], season=[2010, 2010], game_type=["REG", "REG"], week=[2, 1],
        gameday=["2010-09-19", "2010-09-12"], away_team=["OAK", "SD"], home_team=["STL", "KC"],
        away_score=[10, 20], home_score=[17, 21], result=[7, 1], location=["Home", "Neutral"],
        away_rest=[7, 7], home_rest=[7, 7], away_moneyline=[150, -110], home_moneyline=[-170, -110],
        spread_line=[3.5, 0.5], total_line=[40, 41], div_game=[0, 1], roof=["dome", "outdoors"],
        away_qb_id=["q1", "q2"], home_qb_id=["q3", "q4"], away_qb_name=["", ""], home_qb_name=["", ""]))
    g = clean_games(raw, 2010, 2010)
    assert list(g["game_id"]) == ["x1", "x2"]
    assert set(g["home_team"]) | set(g["away_team"]) == {"LV", "LAC", "LA", "KC"}
    assert list(g["neutral"]) == [1, 0]
    assert set(FRANCHISE.values()) == {"LV", "LAC", "LA"}
