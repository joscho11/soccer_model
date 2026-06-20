"""Live-tournament betting workflow: slate board, odds logging, grading, report.

The model is calibrated; the missing piece is the market. This turns predictions
into a 30-second daily loop and — critically — logs every bet's (model_prob, odds,
result) so that by the end of the tournament you have a real record to measure
edge *against the line*, not just against base rates.

    python src/slate.py board                       # today+tomorrow fixtures w/ probs & fair odds
    python src/slate.py log England Croatia --market home --odds 1.95
    python src/slate.py log England Croatia --market over --line 2.5 --odds 2.10
    python src/slate.py close England Croatia --market home --odds 1.80   # record closing line (CLV)
    python src/slate.py grade                        # fill results from played games
    python src/slate.py report                       # record, ROI, calibration, CLV
    python src/slate.py open                          # show ungraded bets

Flat 1-unit stakes by default. Bet log lives at data/bet_log.csv.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from dixon_coles import load_results
from predict import load_model
import markets

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "bet_log.csv"
LOG_COLS = ["logged_at", "date", "home_team", "away_team", "market", "line",
            "model_prob", "odds", "implied_prob", "edge", "ev_per_unit", "stake",
            "tier", "closing_odds", "result", "profit_units"]
MARKETS = ("home", "draw", "away", "over", "under", "btts_yes", "btts_no",
           "ah_home", "ah_away")
# markets that carry a numeric line (and so need it for dedupe / grading / display)
LINE_MARKETS = ("over", "under", "ah_home", "ah_away")

# Confidence tiers by model-vs-market edge (probability points), the soccer analog
# of the NFL "voters agree + edge >= 3pt" rule. PASS = don't bet. Sized only after
# a tier's forward CLV proves out (see kelly.py / report's by-tier CLV).
TIER_HIGH = 0.05
TIER_MEDIUM = 0.03


def tier_for(edge: float) -> str:
    """HIGH/MEDIUM/PASS from a model-vs-market edge in probability points."""
    if edge >= TIER_HIGH:
        return "HIGH"
    if edge >= TIER_MEDIUM:
        return "MEDIUM"
    return "PASS"


def _pct(p: float) -> str:
    return f"{100 * p:5.1f}%"


def _line_sfx(market: str, line) -> str:
    return f" {line}" if market in LINE_MARKETS else ""


# ---- model probability for a market -------------------------------------
def market_prob(model, home: str, away: str, market: str, line: float,
                neutral: bool) -> float:
    if market in ("home", "draw", "away"):
        o = markets.outcome_probs(model.score_matrix(home, away, neutral=neutral))
        return float(o[market])
    if market in ("btts_yes", "btts_no"):
        b = markets.btts(model.score_matrix(home, away, neutral=neutral))
        return float(b[market])
    if market in ("ah_home", "ah_away"):
        # `line` is the signed handicap on the chosen team (e.g. ah_home -1.5).
        # asian_handicap(mat, h) takes the *home* handicap h and returns the away
        # side at -h, so we flip the sign for the away bet.
        mat = model.score_matrix(home, away, neutral=neutral)
        if market == "ah_home":
            return float(markets.asian_handicap(mat, line)[f"home_{line:+g}"])
        return float(markets.asian_handicap(mat, -line)[f"away_{line:+g}"])
    over = model.over_prob(home, away, neutral=neutral, line=line, calibrated=True)
    return float(over if market == "over" else 1 - over)


def find_fixture(results: pd.DataFrame, home: str, away: str):
    """Return (date, neutral) for the match: the earliest *unplayed* fixture for
    these teams (the normal pre-kickoff case), else the most recent played one
    (so grading still works if you log after the fact). Falls back to (today, True)."""
    m = results[(results["home_team"] == home) & (results["away_team"] == away)]
    if m.empty:
        return date.today(), True
    unplayed = m[m["home_score"].isna()].sort_values("date")
    row = unplayed.iloc[0] if not unplayed.empty else m.sort_values("date").iloc[-1]
    return row["date"].date(), bool(row["neutral"])


# ---- board ---------------------------------------------------------------
def board_cmd(args) -> None:
    model = load_model()
    results = load_results()
    results["date"] = pd.to_datetime(results["date"])
    results["neutral"] = results["neutral"].astype(str).str.upper().eq("TRUE")
    start = pd.Timestamp(args.start) if args.start else pd.Timestamp(date.today())
    end = start + pd.Timedelta(days=args.days)
    up = results[results["home_score"].isna() & (results["date"] >= start)
                 & (results["date"] < end)]
    if args.tournament:
        up = up[up["tournament"].str.contains(args.tournament, case=False, na=False)]
    up = up.sort_values("date")
    if up.empty:
        print(f"No unplayed fixtures in {start.date()} .. {(end - pd.Timedelta(days=1)).date()}.")
        return
    cal = "recalibrated" if model.totals_calib else "raw"
    print(f"Slate {start.date()} .. {(end - pd.Timedelta(days=1)).date()}  "
          f"(neutral venue, O/U {cal})\n")
    print(f"{'date':11s} {'match':32s} {'home':>13} {'draw':>13} {'away':>13} {'o2.5':>6}")
    print("-" * 96)
    for _, r in up.iterrows():
        h, a = r["home_team"], r["away_team"]
        try:
            o = markets.outcome_probs(model.score_matrix(h, a, neutral=True))
            over = model.over_prob(h, a, neutral=True, line=2.5, calibrated=True)
        except KeyError:
            continue

        def cell(p):  # prob + fair decimal odds
            return f"{_pct(p)}/{markets.fair_odds(p):>4}"
        print(f"{r['date'].date()!s:11s} {f'{h} v {a}':32.32s} "
              f"{cell(o['home']):>13} {cell(o['draw']):>13} {cell(o['away']):>13} {_pct(over)}")
    print("\nfair = no-margin model odds. `log <home> <away> --market <m> --odds <your line>`")


# ---- log -----------------------------------------------------------------
def _read_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        for c in LOG_COLS:                      # tolerate pre-tier logs
            if c not in df.columns:
                df[c] = ""
        # Normalise `line` to a stable string ("" for non-line markets). CSV turns an
        # all-empty / mixed line column into float64 NaN -> "nan", which would break
        # dedupe (pending_exists) and close-line matching. Single read path = fix once.
        df["line"] = df["line"].fillna("").astype(str).replace("nan", "")
        return df[LOG_COLS]
    return pd.DataFrame(columns=LOG_COLS)


def _write_log(df: pd.DataFrame) -> None:
    df.to_csv(LOG_PATH, index=False)


def pending_exists(home: str, away: str, market: str, line) -> bool:
    """True if an ungraded bet for this selection is already logged (dedupe)."""
    df = _read_log()
    if df.empty:
        return False
    line_str = str(line) if market in LINE_MARKETS else ""
    m = ((df["home_team"] == home) & (df["away_team"] == away)
         & (df["market"] == market) & (df["line"].astype(str) == line_str)
         & (df["result"] == "pending"))
    return bool(m.any())


def record_bet(home: str, away: str, market: str, odds: float, model_prob: float,
               mdate, line: float = 2.5, stake: float = 1.0,
               tier: str | None = None, edge: float | None = None) -> dict:
    """Append one bet to the log and return its value summary. Shared by the
    manual `log` command and the automated odds scanner. `tier` may be passed
    directly (auto-scan, edge vs de-vigged consensus); otherwise it's derived
    from `edge` if given, else from the price-implied edge."""
    v = markets.value_vs_odds(model_prob, odds)
    if tier is None:
        tier = tier_for(edge if edge is not None else v["edge"])
    row = {
        "logged_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "date": str(mdate), "home_team": home, "away_team": away,
        "market": market, "line": line if market in LINE_MARKETS else "",
        "model_prob": round(model_prob, 4), "odds": odds,
        "implied_prob": v["implied_prob"], "edge": v["edge"],
        "ev_per_unit": v["ev_per_unit"], "stake": stake, "tier": tier,
        "closing_odds": "", "result": "pending", "profit_units": "",
    }
    existing = _read_log()
    new = pd.DataFrame([row], columns=LOG_COLS)
    _write_log(new if existing.empty else pd.concat([existing, new], ignore_index=True))
    return v


def log_cmd(args) -> None:
    model = load_model()
    results = load_results()
    results["date"] = pd.to_datetime(results["date"])
    results["neutral"] = results["neutral"].astype(str).str.upper().eq("TRUE")
    if args.market not in MARKETS:
        raise SystemExit(f"--market must be one of {MARKETS}")

    mdate, neutral = find_fixture(results, args.home, args.away)
    if args.home_venue:
        neutral = False
    prob = market_prob(model, args.home, args.away, args.market, args.line, neutral)
    v = record_bet(args.home, args.away, args.market, args.odds, prob, mdate,
                   line=args.line, stake=args.stake)

    flag = "  <== +EV" if v["value"] else "  (no value at this price)"
    sel = args.market + _line_sfx(args.market, args.line)
    print(f"logged: {args.home} v {args.away} [{sel}] @ {args.odds}  "
          f"[{tier_for(v['edge'])} — price-implied edge, not consensus]")
    print(f"  model {_pct(prob)}  implied {_pct(v['implied_prob'])}  "
          f"edge {v['edge']:+.3f}  ev/unit {v['ev_per_unit']:+.3f}{flag}")


def close_cmd(args) -> None:
    """Record the closing line for an already-logged bet, to measure CLV."""
    df = _read_log()
    mask = ((df["home_team"] == args.home) & (df["away_team"] == args.away)
            & (df["market"] == args.market))
    # disambiguate by line for O/U and AH (a fixture can carry several AH lines)
    if args.market in LINE_MARKETS:
        if args.line is None:
            raise SystemExit(f"--line is required to close a {args.market} bet "
                             f"(a fixture may have several lines).")
        mask &= (df["line"].astype(str) == str(args.line))
    if not mask.any():
        raise SystemExit("No matching logged bet. Log it first.")
    df.loc[mask, "closing_odds"] = args.odds
    _write_log(df)
    print(f"closing odds {args.odds} recorded for {args.home} v {args.away} [{args.market}] "
          f"({int(mask.sum())} row(s))")


# ---- grade ---------------------------------------------------------------
def _won(market: str, line, hs: int, as_: int) -> tuple[str, bool | None]:
    total = hs + as_
    if market == "home":
        return ("win" if hs > as_ else "loss"), None
    if market == "away":
        return ("win" if as_ > hs else "loss"), None
    if market == "draw":
        return ("win" if hs == as_ else "loss"), None
    if market == "btts_yes":
        return ("win" if hs > 0 and as_ > 0 else "loss"), None
    if market == "btts_no":
        return ("win" if hs == 0 or as_ == 0 else "loss"), None
    if market in ("ah_home", "ah_away"):
        # cover = (team margin) + handicap > 0; exactly 0 is a push (whole lines only)
        margin = (hs - as_) if market == "ah_home" else (as_ - hs)
        m = margin + float(line)
        return ("push" if m == 0 else ("win" if m > 0 else "loss")), None
    ln = float(line)
    if market == "over":
        return ("push" if total == ln else ("win" if total > ln else "loss")), None
    if market == "under":
        return ("push" if total == ln else ("win" if total < ln else "loss")), None
    raise ValueError(market)


def grade_cmd(args) -> None:
    df = _read_log()
    if df.empty:
        print("Bet log is empty.")
        return
    results = load_results()
    results["date"] = pd.to_datetime(results["date"])
    played = results.dropna(subset=["home_score", "away_score"]).copy()
    played["key"] = (played["home_team"] + "|" + played["away_team"] + "|"
                     + played["date"].dt.strftime("%Y-%m-%d"))
    score_by_key = {k: (int(h), int(a)) for k, h, a in
                    zip(played["key"], played["home_score"], played["away_score"])}

    graded = 0
    for i, r in df.iterrows():
        if str(r["result"]) != "pending":
            continue
        key = f"{r['home_team']}|{r['away_team']}|{r['date']}"
        if key not in score_by_key:
            continue
        hs, as_ = score_by_key[key]
        result, _ = _won(r["market"], r["line"], hs, as_)
        stake = float(r["stake"])
        profit = 0.0 if result == "push" else (stake * (float(r["odds"]) - 1)
                                               if result == "win" else -stake)
        df.at[i, "result"] = result
        df.at[i, "profit_units"] = round(profit, 3)
        graded += 1
    _write_log(df)
    print(f"graded {graded} bet(s); "
          f"{int((df['result'] == 'pending').sum())} still pending (game not played / not in data).")


# ---- report --------------------------------------------------------------
def report_cmd(args) -> None:
    df = _read_log()
    if df.empty:
        print("Bet log is empty. Log some bets first.")
        return
    settled = df[df["result"].isin(["win", "loss", "push"])].copy()
    print(f"=== Bet log report ===  ({len(df)} logged, {len(settled)} settled)\n")
    if settled.empty:
        print("Nothing settled yet -- run `grade` after games are played.")
        return

    settled["profit_units"] = settled["profit_units"].astype(float)
    settled["stake"] = settled["stake"].astype(float)
    wins = (settled["result"] == "win").sum()
    pushes = (settled["result"] == "push").sum()
    decisive = settled[settled["result"] != "push"]
    staked = settled["stake"].sum()
    profit = settled["profit_units"].sum()

    print(f"  record        {wins}W-{len(decisive) - wins}L"
          f"{f'-{pushes}P' if pushes else ''}  "
          f"(hit rate {wins / max(len(decisive), 1) * 100:.1f}%)")
    print(f"  staked        {staked:.1f}u")
    print(f"  profit        {profit:+.2f}u")
    print(f"  ROI           {profit / staked * 100:+.1f}%" if staked else "  ROI    n/a")

    # is the model's EV predictive? (avg modelled edge vs realised win rate)
    print(f"\n  model check:")
    print(f"    avg model prob   {settled['model_prob'].astype(float).mean() * 100:.1f}%"
          f"   realised win rate {wins / max(len(decisive), 1) * 100:.1f}%")
    print(f"    avg ev/unit      {settled['ev_per_unit'].astype(float).mean():+.3f}"
          f"   realised profit/unit {profit / staked:+.3f}" if staked else "")

    # closing-line value, where recorded
    clv = settled.copy()
    clv["closing_odds"] = pd.to_numeric(clv["closing_odds"], errors="coerce")
    clv = clv[clv["closing_odds"].notna()]
    if not clv.empty:
        clv["odds"] = clv["odds"].astype(float)
        beat = (clv["odds"] > clv["closing_odds"]).mean()
        # CLV in prob terms: implied at close minus implied at bet (positive = beat close)
        clv_edge = (1 / clv["closing_odds"] - 1 / clv["odds"]).mean()
        print(f"\n  closing line value  ({len(clv)} bets w/ closing odds):")
        print(f"    beat the close   {beat * 100:.0f}% of the time")
        print(f"    avg CLV          {clv_edge * 100:+.2f} prob-points  "
              f"({'positive = sharp' if clv_edge > 0 else 'negative'})")

    # by tier: the headline honest test — does the HIGH tier clear the close and profit?
    settled["tier"] = settled["tier"].fillna("").replace("", "(none)")
    print(f"\n  by tier (record / ROI / CLV):")
    print(f"    {'tier':8s} {'n':>3} {'hit%':>6} {'ROI':>7} {'n_clv':>5} {'beat%':>6} {'avgCLV':>7}")
    for tier, g in settled.groupby("tier"):
        dec_t = g[g["result"] != "push"]
        w_t = (g["result"] == "win").sum()
        hit = w_t / max(len(dec_t), 1) * 100
        staked_t = g["stake"].sum()
        roi = g["profit_units"].sum() / staked_t * 100 if staked_t else float("nan")
        gc = g.copy()
        gc["closing_odds"] = pd.to_numeric(gc["closing_odds"], errors="coerce")
        gc = gc[gc["closing_odds"].notna()]
        if len(gc):
            gc["odds"] = gc["odds"].astype(float)
            beat_t = (gc["odds"] > gc["closing_odds"]).mean() * 100
            clv_t = (1 / gc["closing_odds"] - 1 / gc["odds"]).mean() * 100
            clv_cells = f"{len(gc):>5} {beat_t:5.0f}% {clv_t:+6.2f}"
        else:
            clv_cells = f"{0:>5} {'—':>6} {'—':>7}"
        print(f"    {tier:8s} {len(g):>3} {hit:5.0f}% {roi:+6.1f}% {clv_cells}")

    by_mkt = settled.groupby("market")["profit_units"].agg(["count", "sum"])
    print(f"\n  by market:")
    for mkt, row in by_mkt.iterrows():
        print(f"    {mkt:8s} {int(row['count']):3d} bets  {row['sum']:+.2f}u")


def open_cmd(args) -> None:
    df = _read_log()
    pend = df[df["result"] == "pending"]
    if pend.empty:
        print("No open bets.")
        return
    print(f"{len(pend)} open bet(s):")
    for _, r in pend.iterrows():
        sel = f"{r['market']}{_line_sfx(r['market'], r['line'])}"
        print(f"  {r['date']}  {r['home_team']} v {r['away_team']} [{sel}] @ {r['odds']}  "
              f"(model {float(r['model_prob'])*100:.1f}%, edge {float(r['edge']):+.3f})")


def main() -> None:
    p = argparse.ArgumentParser(description="Live slate, odds logging & CLV tracking")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("board", help="upcoming fixtures with probs and fair odds")
    b.add_argument("--days", type=int, default=2, help="window length (default today+tomorrow)")
    b.add_argument("--start", help="start date YYYY-MM-DD (default today)")
    b.add_argument("--tournament", help="filter, e.g. 'World Cup'")
    b.set_defaults(func=board_cmd)

    lg = sub.add_parser("log", help="log a bet at a posted price")
    lg.add_argument("home"); lg.add_argument("away")
    lg.add_argument("--market", required=True, help=f"one of {MARKETS}")
    lg.add_argument("--odds", type=float, required=True, help="decimal odds you can bet")
    lg.add_argument("--line", type=float, default=2.5,
                    help="O/U total or AH handicap (e.g. -1.5 for ah_home; default 2.5)")
    lg.add_argument("--stake", type=float, default=1.0)
    lg.add_argument("--home-venue", action="store_true")
    lg.set_defaults(func=log_cmd)

    cl = sub.add_parser("close", help="record closing odds for a logged bet (CLV)")
    cl.add_argument("home"); cl.add_argument("away")
    cl.add_argument("--market", required=True)
    cl.add_argument("--odds", type=float, required=True)
    cl.add_argument("--line", type=float, default=None,
                    help="required for O/U and AH to pick the right line")
    cl.set_defaults(func=close_cmd)

    sub.add_parser("grade", help="fill results from played games").set_defaults(func=grade_cmd)
    sub.add_parser("report", help="record, ROI, calibration, CLV").set_defaults(func=report_cmd)
    sub.add_parser("open", help="show ungraded bets").set_defaults(func=open_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
