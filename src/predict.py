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
    # The on-disk schema persists only the baseline model. The experimental features
    # (elo / squad / context) are backtest-only and NOT serialized, so refuse to save a
    # model fit with them rather than silently round-trip back to a plain model.
    if m.use_elo or m.elo_prior or m.squad_prior or m.use_context:
        raise ValueError("save_model supports the baseline model only; elo/squad/context "
                         "are backtest-only experiments and are not persisted.")
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


# ---- daily slate (predictions only) -------------------------------------
def _ah_confidence(cover_prob: float) -> str:
    """Confidence that the favourite COVERS the posted Asian-handicap line, from how
    far the model's cover probability sits from a 50/50 coin-flip — i.e. how strongly
    the model thinks one side of the book's line is mispriced. This is the useful
    question: 'Spain beats Saudi Arabia' is obvious and the books know it; 'Spain
    covers -2.5' is not. NOT match-winner confidence."""
    d = abs(cover_prob - 0.5)
    if d >= 0.12:
        return "HIGH"
    if d >= 0.06:
        return "MEDIUM"
    return "LOW"


def _odds_enrich(target: str, model) -> tuple[dict, dict]:
    """One Odds API call (via odds._gather, ~3 credits) -> (kickoffs, ah_lines):
      kickoffs[(home, away)] = 'HH:MM UTC (HH:MM ET)'  (ET = UTC-4, EDT, Jun–Jul 2026)
      ah_lines[(home, away)] = list of Asian-handicap rows {market (ah_home/ah_away),
          line (signed handicap on that side), model_prob (model cover for that side),
          mkt_prob (de-vigged implied cover), price} — whole/half lines only (the
          scanner skips quarter-lines, which aren't priced).
    Best-effort: ({}, {}) if no key / offline / lookup fails, so predictions never
    break on the network. ~3 credits per run (reuses the betting scanner's feed)."""
    try:
        import os
        from types import SimpleNamespace
        from datetime import datetime, timezone, timedelta
        import odds  # lazy: odds imports predict, so import here to avoid a cycle
        odds.load_dotenv()
        if not os.environ.get("ODDS_API_KEY"):
            return {}, {}
        _, _, events, matched, _, _ = odds._gather(SimpleNamespace(sport=None, regions="us"))
        cache: dict = {}
        kick: dict = {}
        for ev in events:
            if str(ev.get("commence_time", ""))[:10] != target:
                continue
            h = odds.normalize(ev["home_team"], model.teams, cache)
            a = odds.normalize(ev["away_team"], model.teams, cache)
            if h is None or a is None:
                continue
            dt = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            et = dt.astimezone(timezone(timedelta(hours=-4)))
            kick[(h, a)] = f"{dt:%H:%M} UTC ({et:%H:%M} ET)"
        ah: dict = {}
        for rows in matched:
            for row in rows:
                if row["market"] in ("ah_home", "ah_away"):
                    ah.setdefault((row["home"], row["away"]), []).append(row)
        return kick, ah
    except (Exception, SystemExit):
        return {}, {}


def day_cmd(args) -> None:
    m = load_model()
    results = load_results()
    results["date"] = pd.to_datetime(results["date"])
    target = pd.Timestamp(args.date)
    fixtures = results[(results["tournament"] == "FIFA World Cup")
                       & (results["date"] == target)].sort_values("home_team")
    if fixtures.empty:
        print(f"No World Cup fixtures on {args.date}.")
        up = results[(results["tournament"] == "FIFA World Cup") & results["home_score"].isna()]
        dates = sorted(up["date"].dt.strftime("%Y-%m-%d").unique())[:10]
        if dates:
            print("upcoming fixture dates: " + ", ".join(dates))
        return

    kick, ah = ({}, {}) if args.no_odds else _odds_enrich(args.date, m)

    print(f"\nWorld Cup predictions - {target:%A %B %d, %Y}   ({len(fixtures)} matches)")
    print(f"model fitted on {m.fitted_on}, {m.n_matches:,} matches\n")

    ranking = []
    for _, r in fixtures.iterrows():
        h, a = r["home_team"], r["away_team"]
        neutral = str(r["neutral"]).upper() == "TRUE"
        try:
            mat = m.score_matrix(h, a, neutral=neutral)
            lh, la = m.lambdas(h, a, neutral)
        except KeyError:
            print("=" * 70)
            print(f"{h} vs {a}    kickoff {kick.get((h, a), 'time TBD')}")
            print("  insufficient recent data to predict (team below the model's "
                  "min-matches threshold).\n")
            continue

        o = markets.outcome_probs(mat)
        ou = markets.over_under(mat, 2.5)
        b = markets.btts(mat)
        score, _ = markets.correct_score(mat, 1)[0]
        names = {"home": f"{h} win", "draw": "Draw", "away": f"{a} win"}
        best = max(("home", "draw", "away"), key=lambda k: o[k])
        fav_is_home = o["home"] >= o["away"]
        fav = h if fav_is_home else a

        print("=" * 70)
        print(f"{h} vs {a}    kickoff {kick.get((h, a), 'time TBD')}   "
              f"[{'neutral' if neutral else h + ' at home'}]")
        print(f"  predicted score   {score}   (xG: {h} {lh:.2f} - {la:.2f} {a})")
        print(f"  1X2 (context)  {h} {_pct(o['home'])} (fair {markets.fair_odds(o['home'])})"
              f"   Draw {_pct(o['draw'])} (fair {markets.fair_odds(o['draw'])})"
              f"   {a} {_pct(o['away'])} (fair {markets.fair_odds(o['away'])})")
        print(f"  O/U 2.5           over {_pct(ou['over_2.5'])}   under {_pct(ou['under_2.5'])}")
        print(f"  BTTS              yes {_pct(b['btts_yes'])}   no {_pct(b['btts_no'])}")
        print(f"  most likely       {names[best]} ({_pct(o[best])})")

        # headline: how confident is the model that the FAVOURITE covers the book's
        # main (most balanced) Asian-handicap line — derived from the score matrix,
        # measured against the de-vigged implied cover prob.
        side = "ah_home" if fav_is_home else "ah_away"
        side_rows = [row for row in ah.get((h, a), []) if row["market"] == side]
        if not side_rows:
            print("  AH confidence     n/a   (no whole/half Asian-handicap line in the "
                  "feed for this game)")
        else:
            main = min(side_rows, key=lambda row: abs(row["mkt_prob"] - 0.5))
            cover, implied, line = main["model_prob"], main["mkt_prob"], main["line"]
            edge = cover - implied
            conf = _ah_confidence(cover)
            print(f"  AH confidence     {conf}   {fav} {line:+g}: model cover "
                  f"{_pct(cover)} vs line-implied {_pct(implied)} (edge {edge*100:+.1f}pp)")
            ranking.append({"fav": fav, "line": line, "cover": cover,
                            "implied": implied, "edge": edge, "conf": conf})
        print()

    if ranking:
        print("=" * 70)
        print("Biggest mismatches - strongest handicap-cover disagreement "
              "(model cover vs posted line):\n")
        for i, c in enumerate(sorted(ranking, key=lambda c: abs(c["edge"]), reverse=True), 1):
            print(f"  {i:>2}. {c['fav']:<18} {c['line']:+g}   model cover {_pct(c['cover'])} "
                  f"vs implied {_pct(c['implied'])}  (edge {c['edge']*100:+.1f}pp)   [{c['conf']}]")
    else:
        print("(No Asian-handicap lines in the feed — re-run with the odds feed "
              "available for the handicap-cover ranking.)")

    print("\nNote: the model prices TEAM goals only - no player props "
          "(goalscorer/shots/cards);\nanything player-level would be fabricated, so it's "
          "omitted. Confidence is Asian-handicap\ncover confidence (model vs the posted "
          "line), not match-winner or betting edge.")


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

    d = sub.add_parser("day", help="full predictions printout for one date's WC slate")
    d.add_argument("date", help="match date YYYY-MM-DD")
    d.add_argument("--no-odds", action="store_true",
                   help="skip the odds-feed lookup (no kickoff times, no Asian-handicap "
                        "cover confidence); fully offline")
    d.set_defaults(func=day_cmd)

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
