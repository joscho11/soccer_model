"""Point-in-time squad-value index from the (free) salimt Transfermarkt datalake.

For each national team and date d, the index is the log of the summed market value of
its top-N most valuable players who have a recent value observation as of d — a proxy
for "how good is the available talent," computed point-in-time (only value datapoints
on or before d). This is genuinely NEW information vs. international results (it's club
market value), which is why it's worth testing as a strength prior.

Squad membership is approximated by player CITIZENSHIP (the datalake has no per-match
call-ups), so diaspora nations (Cape Verde, Curaçao, ...) are undercounted and the
small/Gulf nations are sparsely valued — a known limitation (see the A1 coverage check).

Data is downloaded once and cached under data/ (gitignored).
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE = "https://raw.githubusercontent.com/salimt/football-datasets/main/datalake/transfermarkt"
FILES = {
    "tm_market_value.csv": f"{BASE}/player_market_value/player_market_value.csv",
    "tm_profiles.csv": f"{BASE}/player_profiles/player_profiles.csv",
}

# martj42 team name -> Transfermarkt citizenship string, where they differ.
ALIASES = {
    "Ivory Coast": "Cote d'Ivoire", "South Korea": "Korea, South",
    "North Korea": "Korea, North", "Republic of Ireland": "Ireland",
}

_CACHE: pd.DataFrame | None = None


def _ensure_data() -> None:
    for name, url in FILES.items():
        path = DATA_DIR / name
        if path.exists():
            continue
        print(f"downloading {name} ...")
        with urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())


def _load() -> pd.DataFrame:
    """Market-value history with each player's citizenship attached (cached)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    _ensure_data()
    prof = pd.read_csv(DATA_DIR / "tm_profiles.csv",
                       usecols=["player_id", "citizenship"], low_memory=False)
    mv = pd.read_csv(DATA_DIR / "tm_market_value.csv", low_memory=False)
    date_col = "date_unix" if "date_unix" in mv.columns else "date"
    mv = mv.rename(columns={date_col: "date"})[["player_id", "date", "value"]]
    mv["date"] = pd.to_datetime(mv["date"], errors="coerce")
    mv["value"] = pd.to_numeric(mv["value"], errors="coerce")
    mv = mv.dropna(subset=["date", "value"])
    _CACHE = mv.merge(prof, on="player_id", how="left")
    return _CACHE


def squad_index(teams: list[str], as_of, top_n: int = 23,
                window_days: int = 550) -> dict[str, float]:
    """{team: log1p(sum of top-N players' latest value as of `as_of`)} for teams whose
    citizenship is found. Point-in-time: only value points in (as_of-window, as_of]."""
    mv = _load()
    cutoff = pd.Timestamp(as_of)
    lo = cutoff - pd.Timedelta(days=window_days)
    d = mv[(mv["date"] <= cutoff) & (mv["date"] > lo) & (mv["value"] > 0)]
    if d.empty:
        return {}
    # latest value per player within the window
    d = d.sort_values("date").drop_duplicates("player_id", keep="last")
    by_cz = {cz: g for cz, g in d.groupby("citizenship")}
    out: dict[str, float] = {}
    for t in teams:
        cz = ALIASES.get(t, t)
        g = by_cz.get(cz)
        if g is None or g.empty:
            continue
        topsum = g["value"].nlargest(top_n).sum()
        if topsum > 0:
            out[t] = float(np.log1p(topsum))
    return out


if __name__ == "__main__":
    import sys
    as_of = sys.argv[1] if len(sys.argv) > 1 else "2026-06-01"
    idx = squad_index(["Brazil", "Spain", "Argentina", "France", "Iraq",
                       "Cape Verde", "Uzbekistan", "Norway", "Japan"], as_of)
    print(f"squad-value index (log €) as of {as_of}:")
    for t, v in sorted(idx.items(), key=lambda kv: -kv[1]):
        print(f"  {t:14} {v:6.2f}")
