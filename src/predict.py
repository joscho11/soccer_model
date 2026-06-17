"""CLI for the international football model.

Examples:
    python src/predict.py fit                       # fit and cache the model
    python src/predict.py match "Brazil" "Croatia"  # full market report
    python src/predict.py match France Senegal --odds-home 1.95
    python src/predict.py fixtures                   # predict upcoming WC 2026 games
    python src/predict.py rankings
    python src/predict.py teams Bra                  # search team names
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from dixon_coles import DixonColes, load_results
import markets

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.json"


# ---- persistence ---------------------------------------------------------
def save_model(m: DixonColes) -> None:
    MODEL_PATH.write_text(json.dumps({
        "attack": m.attack, "defence": m.defence, "home_adv": m.home_adv,
        "rho": m.rho,
        "totals_calib": list(m.totals_calib) if m.totals_calib else None,
        "teams": m.teams, "fitted_on": m.fitted_on, "n_matches": m.n_matches,
        "half_life_days": m.half_life_days,
    }, indent=2))


def load_model() -> DixonColes:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No cached model. Run `python src/predict.py fit` first.")
    d = json.loads(MODEL_PATH.read_text())
    m = DixonColes(half_life_days=d.get("half_life_days", 730.0))
    m.attack, m.defence = d["attack"], d["defence"]
    m.home_adv, m.rho = d["home_adv"], d["rho"]
    tc = d.get("totals_calib")
    m.totals_calib = tuple(tc) if tc else None
    m.teams, m.fitted_on, m.n_matches = d["teams"], d["fitted_on"], d["n_matches"]
    return m


def fit_cmd(args) -> None:
    df = load_results()
    m = DixonColes(half_life_days=args.half_life, min_matches=args.min_matches)
    print("fitting Dixon-Coles ...")
    m.fit(df, as_of=date.today())
    save_model(m)
    print(f"fitted on {m.n_matches:,} matches, {len(m.teams)} teams "
          f"(home_adv={m.home_adv:.3f}, rho={m.rho:.3f})")
    print(f"cached -> {MODEL_PATH}")


# ---- pretty printers -----------------------------------------------------
def _pct(p: float) -> str:
    return f"{100 * p:5.1f}%"


def print_match(m: DixonColes, home: str, away: str, neutral: bool, odds: dict) -> None:
    mat = m.score_matrix(home, away, neutral=neutral)
    rep = markets.market_report(mat)
    lh, la = m.lambdas(home, away, neutral)
    print(f"\n{home} vs {away}  ({'neutral' if neutral else home + ' at home'})")
    print(f"expected goals: {home} {lh:.2f} - {la:.2f} {away}")
    o = rep["outcome"]
    print(f"\n  1X2     {home} {_pct(o['home'])} (fair {markets.fair_odds(o['home'])})"
          f"   Draw {_pct(o['draw'])} (fair {markets.fair_odds(o['draw'])})"
          f"   {away} {_pct(o['away'])} (fair {markets.fair_odds(o['away'])})")
    over_cal = m.over_prob(home, away, neutral, line=2.5, calibrated=True)
    if m.totals_calib is not None:
        over_raw = m.over_prob(home, away, neutral, line=2.5, calibrated=False)
        print(f"  O/U 2.5 over {_pct(over_cal)}   under {_pct(1 - over_cal)}"
              f"   (recalibrated; raw Poisson {_pct(over_raw)})")
    else:
        print(f"  O/U 2.5 over {_pct(over_cal)}   under {_pct(1 - over_cal)}"
              f"   (uncalibrated; run calibrate.py)")
    b = rep["btts"]
    print(f"  BTTS    yes {_pct(b['btts_yes'])}   no {_pct(b['btts_no'])}")
    print("  top correct scores: " +
          "  ".join(f"{s} {_pct(p)}" for s, p in rep["correct_score_top6"]))

    # optional value check against posted odds
    checks = []
    if odds.get("home"):
        checks.append(("home win", o["home"], odds["home"]))
    if odds.get("draw"):
        checks.append(("draw", o["draw"], odds["draw"]))
    if odds.get("away"):
        checks.append(("away win", o["away"], odds["away"]))
    if odds.get("over"):
        checks.append(("over 2.5", over_cal, odds["over"]))
    if checks:
        print("\n  value vs posted odds:")
        for name, prob, dec in checks:
            v = markets.value_vs_odds(prob, dec)
            flag = "  <== +EV" if v["value"] else ""
            print(f"    {name:9s} @ {dec:>5}  edge {v['edge']:+.3f}  "
                  f"ev/unit {v['ev_per_unit']:+.3f}{flag}")


def match_cmd(args) -> None:
    m = load_model()
    odds = {"home": args.odds_home, "draw": args.odds_draw,
            "away": args.odds_away, "over": args.odds_over}
    print_match(m, args.home, args.away, neutral=not args.home_venue, odds=odds)


def fixtures_cmd(args) -> None:
    m = load_model()
    df = load_results()
    df["date"] = pd.to_datetime(df["date"])
    upcoming = df[(df["tournament"] == "FIFA World Cup") & df["home_score"].isna()]
    upcoming = upcoming.sort_values("date").head(args.limit)
    if upcoming.empty:
        print("No unplayed World Cup fixtures found in the dataset.")
        return
    print(f"Upcoming World Cup fixtures (model probabilities, neutral venue):\n")
    print(f"{'date':12s} {'match':34s} {'home':>6} {'draw':>6} {'away':>6} {'o2.5':>6}")
    print("-" * 74)
    for _, r in upcoming.iterrows():
        h, a = r["home_team"], r["away_team"]
        try:
            mat = m.score_matrix(h, a, neutral=True)
        except KeyError:
            continue
        o = markets.outcome_probs(mat)
        ou = markets.over_under(mat, 2.5)
        match = f"{h} v {a}"
        print(f"{r['date'].date()!s:12s} {match:34.34s} "
              f"{_pct(o['home'])} {_pct(o['draw'])} {_pct(o['away'])} {_pct(ou['over_2.5'])}")


def rankings_cmd(args) -> None:
    m = load_model()
    print(f"Top {args.top} teams by rating (model fitted on {m.fitted_on}):\n")
    print(m.rankings(args.top).to_string(index=False,
          formatters={"attack": "{:+.2f}".format, "defence": "{:+.2f}".format,
                      "rating": "{:+.2f}".format}))


def teams_cmd(args) -> None:
    m = load_model()
    q = (args.query or "").lower()
    hits = [t for t in m.teams if q in t.lower()]
    print(f"{len(hits)} team(s):")
    print("  " + "\n  ".join(hits))


def main() -> None:
    p = argparse.ArgumentParser(description="International football score & market model")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit and cache the model")
    f.add_argument("--half-life", type=float, default=730.0, help="time-decay half-life in days")
    f.add_argument("--min-matches", type=int, default=8)
    f.set_defaults(func=fit_cmd)

    mt = sub.add_parser("match", help="market report for one fixture")
    mt.add_argument("home")
    mt.add_argument("away")
    mt.add_argument("--home-venue", action="store_true", help="home team plays at home (default neutral)")
    mt.add_argument("--odds-home", type=float)
    mt.add_argument("--odds-draw", type=float)
    mt.add_argument("--odds-away", type=float)
    mt.add_argument("--odds-over", type=float, help="posted decimal odds for over 2.5")
    mt.set_defaults(func=match_cmd)

    fx = sub.add_parser("fixtures", help="predict upcoming World Cup fixtures")
    fx.add_argument("--limit", type=int, default=20)
    fx.set_defaults(func=fixtures_cmd)

    r = sub.add_parser("rankings", help="team strength rankings")
    r.add_argument("--top", type=int, default=20)
    r.set_defaults(func=rankings_cmd)

    t = sub.add_parser("teams", help="search available team names")
    t.add_argument("query", nargs="?", default="")
    t.set_defaults(func=teams_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
