import numpy as np
import pandas as pd
import pytest


def make_games(n_seasons=3, weeks=6, seed=0, teams=("A", "B", "C", "D")):
    """Small synthetic league in the processed-data format."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(2010, 2010 + n_seasons):
        for w in range(1, weeks + 1):
            order = rng.permutation(teams)
            for h, a in zip(order[::2], order[1::2]):
                res = int(round(rng.normal(2, 13)))
                rows.append(dict(game_id=f"{s}_{w}_{a}_{h}", season=s, week=w, game_type="REG",
                                 home_team=h, away_team=a, result=res, neutral=int(rng.random() < 0.1),
                                 home_rest=7, away_rest=int(rng.choice([7, 14])),
                                 home_qb_change=int(rng.random() < 0.1), away_qb_change=int(rng.random() < 0.1),
                                 spread_line=round(rng.normal(0, 4)) + 0.5,
                                 home_qb_id=f"{h}1", away_qb_id=f"{a}1"))
    return pd.DataFrame(rows)


@pytest.fixture
def toy_games():
    return make_games()
