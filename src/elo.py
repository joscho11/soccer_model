"""Point-in-time World Football Elo ratings, computed from the match results we
already have (the same algorithm eloratings.net uses).

Why compute rather than scrape: a no-leakage backtest needs each team's rating *as
of* every historical match date. Recomputing chronologically from results.csv gives
exactly that for free — no scraping, no fragile HTML parsing, and by construction a
rating only ever reflects matches that happened before it.

Algorithm (World Football Elo): R' = R + K*(W - We), where
  We = 1 / (10^(-dr/400) + 1),  dr = (R_home - R_away) + home_adv (home_adv=0 at neutral)
  W  = 1 win / 0.5 draw / 0 loss
  K  = K0(tournament) * G,  G = 1 (|gd|<=1), 1.5 (|gd|=2), (11+|gd|)/8 (|gd|>=3)
The update is zero-sum (home gains exactly what away loses), so ratings stay centred.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# K0 by competition importance (standard World Football Elo weights).
ELO_K = {
    "FIFA World Cup": 60,
    "Copa América": 50, "UEFA Euro": 50, "African Cup of Nations": 50,
    "AFC Asian Cup": 50, "Gold Cup": 50, "CONCACAF Championship": 50,
    "Confederations Cup": 50, "Oceania Nations Cup": 50,
    "UEFA Nations League": 40, "CONCACAF Nations League": 40,
    "Friendly": 20,
}
DEFAULT_K = 30           # other minor tournaments
QUALIFIER_K = 40         # any "... qualification"
START_RATING = 1500.0
HOME_ADV = 100.0


def k_for(tournament: str) -> float:
    if tournament in ELO_K:
        return ELO_K[tournament]
    if isinstance(tournament, str) and "qualification" in tournament.lower():
        return QUALIFIER_K
    return DEFAULT_K


def attach(df: pd.DataFrame, as_of=None) -> tuple[pd.DataFrame, dict]:
    """Add `home_elo_pre`/`away_elo_pre` (each team's rating *before* the match) to a
    copy of `df`, computed chronologically over all played matches up to `as_of`, and
    return (df_with_cols, current_ratings). Point-in-time: pre-match ratings never see
    the match itself or any later one. Unplayed / post-as_of rows get NaN (and are
    dropped downstream by the model's _prepare)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["home_elo_pre"] = np.nan
    df["away_elo_pre"] = np.nan

    sub = df.dropna(subset=["home_score", "away_score"])
    if as_of is not None:
        sub = sub[sub["date"] <= pd.Timestamp(as_of)]
    sub = sub.sort_values("date")

    ratings: dict[str, float] = {}
    idxs, hpre, apre = [], [], []
    for r in sub.itertuples():
        rh = ratings.get(r.home_team, START_RATING)
        ra = ratings.get(r.away_team, START_RATING)
        idxs.append(r.Index)
        hpre.append(rh)
        apre.append(ra)
        neutral = str(r.neutral).upper() == "TRUE"
        dr = (rh - ra) + (0.0 if neutral else HOME_ADV)
        we = 1.0 / (10 ** (-dr / 400.0) + 1.0)
        w = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        gd = abs(int(r.home_score) - int(r.away_score))
        g = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)
        delta = k_for(r.tournament) * g * (w - we)
        ratings[r.home_team] = rh + delta
        ratings[r.away_team] = ra - delta

    if idxs:
        df.loc[idxs, "home_elo_pre"] = hpre
        df.loc[idxs, "away_elo_pre"] = apre
    return df, ratings
