"""Point-in-time, leak-free CONTEXT features — information NOT in the scoreline.

Everything here is knowable strictly before kickoff and is computed only from matches
that happened before date d (rest/congestion) or from fixed geography (travel/climate),
so adding it to the walk-forward backtest introduces no leakage.

Features attached per match, all framed as a home-minus-away ADVANTAGE (so a positive
value should, if the effect is real, favour the home/first team — but the sign is left
for the model to learn):

  rest_adv     home_rest_days  - away_rest_days        (more rest than the opponent)
  cong_adv     away_congestion - home_congestion       (opponent more fixture-congested)
  travel_adv   away_travel_km  - home_travel_km         (opponent further from home)
  climate_adv  away_climate    - home_climate           (opponent further from its latitude)

travel/climate use COUNTRY-centroid coordinates (data/country_coords.csv) as each team's
home and the host country's location — a continental-scale proxy, not exact venues (true
per-venue altitude needs venue elevation we can't source cleanly; see the A1/Phase-A note).
"""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONTEXT_COLS = ["rest_adv", "cong_adv", "travel_adv", "climate_adv"]

# martj42 country name -> name in data/country_coords.csv, where they differ.
_COORD_ALIASES = {
    "Ivory Coast": "Côte d'Ivoire", "DR Congo": "Congo [DRC]",
    "Republic of Ireland": "Ireland",
}
_COORDS: dict[str, tuple[float, float]] | None = None


def _coords() -> dict[str, tuple[float, float]]:
    global _COORDS
    if _COORDS is None:
        df = pd.read_csv(DATA_DIR / "country_coords.csv")
        _COORDS = {r["name"]: (float(r["latitude"]), float(r["longitude"]))
                   for _, r in df.iterrows()}
    return _COORDS


def _latlon(country: str):
    c = _coords()
    name = _COORD_ALIASES.get(country, country)
    return c.get(name)


def _haversine(a, b) -> float:
    (la1, lo1), (la2, lo2) = a, b
    la1, lo1, la2, lo2 = map(radians, (la1, lo1, la2, lo2))
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


def attach(df: pd.DataFrame, cong_days: int = 21, rest_cap: int = 10) -> pd.DataFrame:
    """Return a copy of df with the CONTEXT_COLS added, computed point-in-time."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    order = df.sort_values("date").index

    last: dict[str, pd.Timestamp] = {}       # team -> date of previous match
    recent: dict[str, list] = {}             # team -> recent match dates (for congestion)
    hr, ar, hc, ac = {}, {}, {}, {}          # per-index rest/congestion for home/away
    for i in order:
        d = df.at[i, "date"]
        for team, rest_d, cong_d in ((df.at[i, "home_team"], hr, hc),
                                     (df.at[i, "away_team"], ar, ac)):
            prev = last.get(team)
            rest_d[i] = (d - prev).days if prev is not None else np.nan
            win = [x for x in recent.get(team, []) if (d - x).days <= cong_days]
            cong_d[i] = len(win)
            recent[team] = win + [d]
            last[team] = d

    df["home_rest"] = pd.Series(hr)
    df["away_rest"] = pd.Series(ar)
    rest_adv = (df["home_rest"].clip(upper=rest_cap) - df["away_rest"].clip(upper=rest_cap))
    df["rest_adv"] = rest_adv.fillna(0.0)
    df["cong_adv"] = (pd.Series(ac) - pd.Series(hc)).fillna(0.0)

    # travel / climate from country centroids and the host country (df["country"])
    travel_adv, climate_adv = [], []
    for i in df.index:
        host = _latlon(df.at[i, "country"])
        home = _latlon(df.at[i, "home_team"])
        away = _latlon(df.at[i, "away_team"])
        if host is None or home is None or away is None:
            travel_adv.append(0.0)
            climate_adv.append(0.0)
            continue
        ht = _haversine(home, host) / 1000.0
        at = _haversine(away, host) / 1000.0
        travel_adv.append(at - ht)                       # away further -> home advantage
        climate_adv.append(abs(host[0] - away[0]) - abs(host[0] - home[0]))  # latitude gap
    df["travel_adv"] = travel_adv
    df["climate_adv"] = climate_adv
    return df


if __name__ == "__main__":
    from dixon_coles import load_results
    d = attach(load_results())
    played = d.dropna(subset=["home_score", "away_score"])
    print("context features attached. summary over played matches:\n")
    print(played[CONTEXT_COLS].describe().round(2).to_string())
