"""Daily $25 -> $100 challenge tracker.

Logs the recommended ticket for a day's World Cup slate and grades it from live
scores, reporting whether $25 would have reached $100. A ticket is a parlay of one
or more legs (a single = a 1-leg parlay): it wins only if no leg loses, and a pushed
leg is dropped from the parlay (its odds factor becomes 1). This is -EV fun against a
sharp market, NOT financial advice (see memory soccer-25-to-100-challenge).

    python src/challenge.py log --date 2026-06-22 --stake 25 --legs '<json>'
    python src/challenge.py grade        # settle from played games (incl. --no-live to skip the feed)
    python src/challenge.py report       # per-day: did $25 reach $100?

legs json: [{"home","away","market","line","odds","model_prob"}, ...]
market uses slate codes (home/draw/away/over/under/btts_yes/btts_no/ah_home/ah_away).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dixon_coles import load_results
from predict import load_model
import slate
import track

LOG = Path(__file__).resolve().parent.parent / "data" / "challenge_log.csv"
COLS = ["date", "ticket_id", "leg", "home", "away", "market", "line", "odds",
        "model_prob", "stake", "ticket_odds", "leg_result", "ticket_result", "payout"]


def _read() -> pd.DataFrame:
    if LOG.exists():
        df = pd.read_csv(LOG)
        for c in COLS:
            if c not in df.columns:
                df[c] = ""
        return df[COLS]
    return pd.DataFrame(columns=COLS)


def _write(df: pd.DataFrame) -> None:
    df.to_csv(LOG, index=False)


def log_cmd(args) -> None:
    legs = json.loads(args.legs)
    ticket_odds = 1.0
    for l in legs:
        ticket_odds *= float(l["odds"])
    df = _read()
    # unique per date even if rows get hand-edited (count distinct tickets, not rows)
    tid = f"{args.date}#{0 if df.empty else df['ticket_id'].nunique()}"
    rows = []
    for i, l in enumerate(legs, 1):
        rows.append({
            "date": args.date, "ticket_id": tid, "leg": i,
            "home": l["home"], "away": l["away"], "market": l["market"],
            "line": l.get("line", ""), "odds": float(l["odds"]),
            "model_prob": l.get("model_prob", ""), "stake": args.stake,
            "ticket_odds": round(ticket_odds, 3), "leg_result": "pending",
            "ticket_result": "pending", "payout": "",
        })
    add = pd.DataFrame(rows, columns=COLS)
    _write(add if df.empty else pd.concat([df, add], ignore_index=True))
    print(f"logged ticket {tid}: {len(legs)} leg(s), stake ${args.stake:g}, combined "
          f"odds {ticket_odds:.2f}  ->  ${args.stake * ticket_odds:.2f} if it hits "
          f"({'reaches' if args.stake * ticket_odds >= 100 else 'short of'} $100).")


def _scores(use_live: bool) -> dict:
    """{(home, away): (hs, as)} from the dataset, plus live Odds API scores."""
    res = load_results()
    res["date"] = pd.to_datetime(res["date"])
    played = res.dropna(subset=["home_score", "away_score"])
    pair = {(h, a): (int(hs), int(a_)) for h, a, hs, a_ in
            zip(played["home_team"], played["away_team"],
                played["home_score"], played["away_score"])}
    if use_live:
        pair.update(track._live_scores(load_model().teams))
    return pair


def grade_cmd(args) -> None:
    df = _read()
    if df.empty:
        print("No challenge tickets logged yet.")
        return
    pair = _scores(use_live=not args.no_live)
    for c in ("leg_result", "ticket_result", "payout"):
        df[c] = df[c].astype(object)

    graded = 0
    for tid, g in df.groupby("ticket_id"):
        if (g["ticket_result"] != "pending").any():
            continue
        legs = []
        for _, r in g.iterrows():
            sc = pair.get((r["home"], r["away"]))
            if sc is None:
                legs = None
                break                       # a leg's game isn't settled yet -> wait
            res, _ = slate._won(r["market"], r["line"], sc[0], sc[1])
            legs.append((r.name, res))
        if legs is None:
            continue
        stake = float(g["stake"].iloc[0])
        lost = any(res == "loss" for _, res in legs)
        payout = 0.0
        if not lost:                         # all win/push: push legs drop (odds factor 1)
            odds = 1.0
            for (i, res), (_, row) in zip(legs, g.iterrows()):
                if res == "win":
                    odds *= float(row["odds"])
            payout = stake * odds
        tres = "loss" if lost else "win"
        for i, res in legs:
            df.at[i, "leg_result"] = res
        df.loc[g.index, "ticket_result"] = tres
        df.loc[g.index, "payout"] = round(payout, 2)
        graded += 1
    _write(df)
    print(f"graded {graded} ticket(s); "
          f"{int((df.groupby('ticket_id')['ticket_result'].first() == 'pending').sum())} "
          f"still pending.")


def report_cmd(args) -> None:
    df = _read()
    if df.empty:
        print("No challenge tickets logged yet.")
        return
    print("=== $25 -> $100 challenge ===\n")
    print(f"  {'date':11} {'legs':>4} {'stake':>6} {'odds':>7} {'result':>8} {'payout':>8}  $100?")
    settled = wins = hit100 = 0
    staked = returned = 0.0
    for tid, g in df.groupby("ticket_id"):
        r0 = g.iloc[0]
        stake = float(r0["stake"])
        res = r0["ticket_result"]
        date = r0["date"]
        if res == "pending":
            print(f"  {date:11} {len(g):>4} {stake:>6.0f} {float(r0['ticket_odds']):>7.2f} "
                  f"{'pending':>8} {'-':>8}")
            continue
        payout = float(r0["payout"])
        settled += 1
        staked += stake
        returned += payout
        wins += res == "win"
        got100 = payout >= 100
        hit100 += got100
        print(f"  {date:11} {len(g):>4} {stake:>6.0f} {float(r0['ticket_odds']):>7.2f} "
              f"{res:>8} {payout:>8.2f}  {'YES' if got100 else 'no'}")
    if settled:
        print(f"\n  settled {settled}: {wins} won, reached $100 on {hit100}.")
        print(f"  staked ${staked:.0f}, returned ${returned:.2f}  (net ${returned - staked:+.2f}).")
    print("\n  -EV fun, not financial advice; the model's picks lean into its known bias.")


def main() -> None:
    p = argparse.ArgumentParser(description="$25 -> $100 daily challenge tracker")
    sub = p.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log", help="log a ticket (parlay of 1+ legs)")
    lg.add_argument("--date", required=True)
    lg.add_argument("--stake", type=float, default=25.0)
    lg.add_argument("--legs", required=True, help="JSON list of legs")
    lg.set_defaults(func=log_cmd)
    gr = sub.add_parser("grade", help="settle tickets from played games")
    gr.add_argument("--no-live", action="store_true", help="dataset only, skip the Odds API feed")
    gr.set_defaults(func=grade_cmd)
    sub.add_parser("report", help="per-day results").set_defaults(func=report_cmd)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
