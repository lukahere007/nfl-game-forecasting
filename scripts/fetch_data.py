"""Download the nflverse games file and write the cleaned parquet.

    python scripts/fetch_data.py

Raw file  -> data/raw/games.csv           (git-ignored)
Processed -> data/processed/games.parquet (committed, ~0.5 MB)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from nflstate.data import GAMES_URL, clean_games, load_raw

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "games.csv"
OUT = ROOT / "data" / "processed" / "games.parquet"


def main() -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {GAMES_URL}")
    urllib.request.urlretrieve(GAMES_URL, RAW)
    games = clean_games(load_raw(RAW))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    games.to_parquet(OUT, index=False)
    print(f"wrote {len(games):,} games ({games.season.min()}-{games.season.max()}) to {OUT}")


if __name__ == "__main__":
    main()
