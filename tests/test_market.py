import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from nflstate.market import american_to_prob, fit_spread_sd, market_prob, overround, remove_vig


def test_american_odds_conversion():
    np.testing.assert_allclose(american_to_prob([-150, 130, -110, 100]), [0.6, 100 / 230, 110 / 210, 0.5])
    with pytest.raises(ValueError):
        american_to_prob(50)


def test_vig_removal_symmetric_line_is_fifty_fifty():
    p = american_to_prob(-110)
    assert remove_vig(p, p) == pytest.approx(0.5)
    assert overround(-110, -110) == pytest.approx(2 * 110 / 210 - 1)  # ~4.76%


def test_vig_removal_known_example_and_sums_to_one():
    ph, pa = american_to_prob(-200), american_to_prob(170)
    assert ph == pytest.approx(2 / 3) and pa == pytest.approx(100 / 270)
    fair_home = remove_vig(ph, pa)
    fair_away = remove_vig(pa, ph)
    assert fair_home == pytest.approx((2 / 3) / (2 / 3 + 100 / 270))
    assert fair_home + fair_away == pytest.approx(1.0)
    assert fair_home < ph  # de-vigging removes the bookmaker's margin


def test_spread_sd_recovered_from_simulation():
    rng = np.random.default_rng(0)
    spread = rng.normal(0, 6, 20000)
    win = rng.random(20000) < norm.cdf(spread / 13.5)
    assert fit_spread_sd(spread, win) == pytest.approx(13.5, rel=0.05)


def test_market_prob_falls_back_to_spread():
    g = pd.DataFrame(dict(home_moneyline=[-200, np.nan], away_moneyline=[170, np.nan], spread_line=[4.5, 3.0]))
    out = market_prob(g, 13.0)
    assert list(out["mkt_source"]) == ["moneyline", "spread"]
    assert out["mkt_p_home"].iloc[1] == pytest.approx(norm.cdf(3 / 13))
