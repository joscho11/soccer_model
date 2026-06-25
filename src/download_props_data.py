"""Fetch + build the inputs the anytime-goalscorer model needs (props.py):

  data/club_players_2025_2026.csv  -- current-season shot volume & minutes (FBref
        top-5 leagues, via the Kaggle mirror hubertsidorowicz/football-players-stats-2025-2026)
  data/understat_players.csv       -- per-player xG / shots history, last 3 seasons,
        top-5 leagues (Kaggle mirror mexwell/understat-database)

Both are gitignored (like results.csv / goalscorers.csv) -- reproducible, not
committed. Needs a Kaggle API token in .env as KAGGLE_API_KEY=<bearer token>;
the new-format bearer token downloads without a username. Stdlib only.

    python src/download_props_data.py
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
CLUB_SLUG = "hubertsidorowicz/football-players-stats-2025-2026"
XG_SLUG = "mexwell/understat-database"
LEAGUES = {"EPL": "EPL", "La_Liga": "La Liga", "Bundesliga": "Bundesliga",
           "Serie_A": "Serie A", "Ligue_1": "Ligue 1"}
XG_MIN_YEAR = 2022      # keep the last three completed seasons


def _key() -> str:
    for base in (Path(__file__).resolve().parent.parent, Path.cwd()):
        env = base / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith("KAGGLE_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    key = os.environ.get("KAGGLE_API_KEY")
    if not key:
        sys.exit("KAGGLE_API_KEY not found in .env or environment.")
    return key


def _download_zip(slug: str, key: str) -> zipfile.ZipFile:
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
    req = Request(url, headers={"Authorization": f"Bearer {key}"})
    print(f"  downloading {slug} ...", flush=True)
    with urlopen(req, timeout=120) as r:
        return zipfile.ZipFile(io.BytesIO(r.read()))


def build_club(key: str) -> None:
    zf = _download_zip(CLUB_SLUG, key)
    name = next(n for n in zf.namelist() if n.endswith("players_data-2025_2026.csv"))
    fb = pd.read_csv(zf.open(name))
    keep = ["Player", "Nation", "Pos", "Squad", "Comp", "MP", "Starts", "Min",
            "Gls", "Sh", "SoT"]
    rename = {}
    for src, dst in [("PK_stats_shooting", "PK"), ("PKatt_stats_shooting", "PKatt")]:
        if src in fb.columns:
            keep.append(src)
            rename[src] = dst
    fb = fb[keep].rename(columns=rename)
    out = DATA / "club_players_2025_2026.csv"
    fb.to_csv(out, index=False, encoding="utf-8")
    print(f"  wrote {out.name}: {len(fb)} players")


def build_xg(key: str) -> None:
    zf = _download_zip(XG_SLUG, key)
    frames = []
    for folder, league in LEAGUES.items():
        member = next(n for n in zf.namelist()
                      if n.endswith(f"{folder}/player.csv"))
        p = pd.read_csv(zf.open(member))
        p = p[p["year"] >= XG_MIN_YEAR].copy()
        p["league"] = league
        frames.append(p[["player_name", "team_title", "year", "league", "time",
                          "games", "goals", "xG", "assists", "xA", "shots", "npg",
                          "npxG", "position"]])
    us = pd.concat(frames, ignore_index=True)
    out = DATA / "understat_players.csv"
    us.to_csv(out, index=False, encoding="utf-8")
    print(f"  wrote {out.name}: {len(us)} player-seasons "
          f"({us['player_name'].nunique()} players, years {sorted(us['year'].unique())})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    key = _key()
    DATA.mkdir(exist_ok=True)
    build_club(key)
    build_xg(key)
    print("done.")


if __name__ == "__main__":
    main()
