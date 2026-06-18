"""Leakage-free retrospective: what would the model have bet, and how did it do?

For every played match in a window, the model is refit on data strictly BEFORE
that match's date (walk-forward, no leakage — the cached model.json can't be used
here because it already contains these results). We then take the model's
high-confidence 1X2 pick and grade it against the actual result.

No bookmaker odds are used: The Odds API's free tier has no historical odds, so we
report hit rate and calibration (did the confident calls come true at the rate
claimed?), not ROI. ROI needs the prices you'd have gotten, which we don't have
for already-played games.

    python src/retro.py --start 2026-06-11 --end 2026-06-17 --min-conf 0.55
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from dixon_coles import DixonColes, load_results
import markets

LABELS = {"home": 0, "draw": 1, "away": 2}


def safe(s: str) -> str:  # Windows cp1252 console can't print e.g. ç
    return s.encode("ascii", "replace").decode("ascii")


def actual_result(hs: int, as_: int) -> str:
    return "home" if hs > as_ else ("draw" if hs == as_ else "away")


def main() -> None:
    p = argparse.ArgumentParser(description="Leakage-free model retrospective")
    p.add_argument("--start", default="2026-06-11")
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--tournament", default="FIFA World Cup")
    p.add_argument("--min-conf", type=float, default=0.55,
                   help="only treat a pick as a 'bet' if model prob >= this")
    p.add_argument("--half-life", type=float, default=730.0)
    p.add_argument("--min-matches", type=int, default=8)
    args = p.parse_args()

    df = load_results()
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    games = df[(df["tournament"] == args.tournament)
               & (df["date"] >= pd.Timestamp(args.start))
               & (df["date"] <= pd.Timestamp(args.end))
               & df["home_score"].notna()].sort_values("date").reset_index(drop=True)
    if games.empty:
        print("No played matches in that window.")
        return

    # walk-forward: one fit per distinct match date, trained strictly before it
    model_cache: dict = {}
    rows = []
    for _, g in games.iterrows():
        d = g["date"]
        if d not in model_cache:
            as_of = (d - pd.Timedelta(days=1)).date()
            model_cache[d] = DixonColes(half_life_days=args.half_life,
                                        min_matches=args.min_matches).fit(df, as_of=as_of)
        m = model_cache[d]
        h, a = g["home_team"], g["away_team"]
        if h not in m.attack or a not in m.attack:
            continue
        o = markets.outcome_probs(m.score_matrix(h, a, neutral=bool(g["neutral"])))
        probs = {"home": o["home"], "draw": o["draw"], "away": o["away"]}
        pick = max(probs, key=probs.get)
        conf = probs[pick]
        hs, as_ = int(g["home_score"]), int(g["away_score"])
        res = actual_result(hs, as_)
        pick_team = {"home": h, "draw": "Draw", "away": a}[pick]
        rows.append({"date": d.date(), "match": f"{safe(h)} v {safe(a)}",
                     "pick": safe(pick_team), "pick_side": pick, "conf": conf,
                     "score": f"{hs}-{as_}", "result": res, "hit": pick == res,
                     "logloss": -np.log(max(probs[res], 1e-12))})

    r = pd.DataFrame(rows)
    bets = r[r["conf"] >= args.min_conf]

    print(f"=== Model retrospective (leakage-free, walk-forward) ===")
    print(f"window {args.start} .. {args.end}   {len(r)} games, "
          f"{len(model_cache)} refits\n")

    print(f"HIGH-CONFIDENCE BETS (model pick prob >= {args.min_conf:.0%}):\n")
    print(f"{'date':11s} {'match':30s} {'pick':16s} {'conf':>5} {'score':>6} {'result':>8}")
    print("-" * 82)
    for _, b in bets.iterrows():
        mark = "WIN " if b["hit"] else "LOSS"
        print(f"{b['date']!s:11s} {b['match']:30.30s} {b['pick']:16.16s} "
              f"{b['conf']*100:4.0f}% {b['score']:>6} {mark:>8}")

    if len(bets):
        w = int(bets["hit"].sum())
        print(f"\n  record         {w}-{len(bets)-w}  (hit rate {w/len(bets)*100:.1f}%)")
        print(f"  avg model conf {bets['conf'].mean()*100:.1f}%  "
              f"<- this is what the model 'expected' to win; compare to hit rate")
        print(f"  calibration    {'OK (close)' if abs(w/len(bets)-bets['conf'].mean())<0.1 else 'OFF'}")

    # context: accuracy of the model's top pick across ALL games (not just confident)
    aw = int(r["hit"].sum())
    print(f"\n  all {len(r)} games, top-pick accuracy {aw/len(r)*100:.1f}%  "
          f"(avg log loss {r['logloss'].mean():.3f})")
    draws = (r["result"] == "draw").sum()
    print(f"  actual results: {(r['result']=='home').sum()} home / {draws} draw / "
          f"{(r['result']=='away').sum()} away")


if __name__ == "__main__":
    main()
