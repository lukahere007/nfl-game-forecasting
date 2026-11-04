"""Loading and cleaning the nflverse ``games.csv`` file.

Everything built here is *pre-game* information: the closing lines, rest days,
venue, and the identity of the starting quarterback (announced before kickoff).
Post-game columns (scores, result) are kept only as targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

#: Relocated franchises are tracked as one team across their history.
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

#: Columns kept in the processed file.
KEEP = [
    "game_id", "season", "game_type", "week", "gameday", "away_team", "home_team",
    "away_score", "home_score", "result", "location", "away_rest", "home_rest",
    "away_moneyline", "home_moneyline", "spread_line", "total_line", "div_game",
    "roof", "away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw nflverse games file."""
    return pd.read_csv(path, low_memory=False)


def clean_games(raw: pd.DataFrame, first_season: int = 2002, last_season: int = 2025) -> pd.DataFrame:
    """Return completed games in ``[first_season, last_season]``, sorted in time.

    * relocated franchises are mapped to their current abbreviation;
    * ``result`` is the home margin (home score - away score), matching the sign
      convention of ``spread_line`` (positive = home favoured);
    * ``neutral`` flags neutral-site games (Super Bowls, most international games);
    * ``home_qb_change`` / ``away_qb_change`` flag a starter who is not the
      team's usual starter so far that season (see :func:`qb_change_flags`).

    2002 is the default start because it is the first 32-team season; seasons
    before the evaluation window serve as burn-in for the ratings.
    """
    g = raw.loc[raw["season"].between(first_season, last_season), KEEP].copy()
    g = g[g["result"].notna()]
    for col in ("home_team", "away_team"):
        g[col] = g[col].replace(FRANCHISE)
    g["gameday"] = pd.to_datetime(g["gameday"])
    g["neutral"] = (g["location"] == "Neutral").astype(int)
    g["playoff"] = (g["game_type"] != "REG").astype(int)
    g = g.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)
    g["home_qb_change"], g["away_qb_change"] = qb_change_flags(g)
    return g


def qb_change_flags(games: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Flag starts by a quarterback other than the team's usual starter.

    The usual starter is the QB with the most starts for that team in *earlier*
    games of the same season (ties broken by the most recent start). A team's
    first game of a season is never flagged, so off-season signings are not
    treated as injuries. Only games strictly before the current one are used.

    ``games`` must be sorted chronologically.
    """
    home = np.zeros(len(games), dtype=int)
    away = np.zeros(len(games), dtype=int)
    history: dict[tuple[int, str], list[str]] = {}
    for i, row in enumerate(games.itertuples(index=False)):
        for side, out in (("home", home), ("away", away)):
            team, qb = getattr(row, f"{side}_team"), getattr(row, f"{side}_qb_id")
            past = history.setdefault((row.season, team), [])
            if past and isinstance(qb, str):
                counts: dict[str, int] = {}
                for q in past:
                    counts[q] = counts.get(q, 0) + 1
                best = max(counts.values())
                usual = next(q for q in reversed(past) if counts[q] == best)
                out[i] = int(qb != usual)
        # update history after both sides are scored (no same-game leakage)
        for side in ("home", "away"):
            qb = getattr(row, f"{side}_qb_id")
            if isinstance(qb, str):
                history[(row.season, getattr(row, f"{side}_team"))].append(qb)
    return home, away


def teams(games: pd.DataFrame) -> list[str]:
    """Sorted list of franchise abbreviations appearing in ``games``."""
    return sorted(set(games["home_team"]) | set(games["away_team"]))
