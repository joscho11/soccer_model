"""Forward tracker for the anytime-goalscorer model.

There is no free historical prop-odds dataset, so the goalscorer model CANNOT be
backtested -- the only honest validation is forward: log the model's probability
and the book's price for every player before kickoff, then grade against who
actually scored. Over a tournament this answers the two questions that matter:

  * Calibration -- when the model says 30%, do ~30% of those players score?
  * Edge -- do the players the model rates ABOVE the book's price actually score
    more often than the price implies (and would backing them have made money)?

      python src/props_track.py freeze --date 2026-06-26          # pull + lock a slate
      python src/props_track.py freeze --event ev.json ECU GER    # offline, saved JSON
      python src/props_track.py close  --date 2026-06-26          # snapshot closing prices
      python src/props_track.py grade                             # fill who scored
      python src/props_track.py report                            # calibration + edge

Record lives at data/props_log.csv (tracked in git -- it's the accuracy record,
like predictions.csv). Penalties count toward a goalscorer; own goals do not.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from dixon_coles import load_results
from predict import load_model
import props

PROPS_PATH = Path(__file__).resolve().parent.parent / "data" / "props_log.csv"
COLS = ["logged_at", "model_fitted_on", "date", "home_team", "away_team", "neutral",
        "player", "model_player", "team", "model_p", "book_odds", "book", "book_p",
        "edge", "closing_odds", "scored", "result"]
GOAL_MARKET = "player_goal_scorer_anytime"
PROPS_REGION = "uk"   # anytime-scorer lives on uk/eu books (Pinnacle, Sky Bet, ...)
WC_START = "2026-06-01"   # grade only against tournament-window games, so a (home,away)
                          # pairing can't collide with a historical friendly in results


def _read() -> pd.DataFrame:
    if PROPS_PATH.exists():
        df = pd.read_csv(PROPS_PATH)
        for c in COLS:
            if c not in df.columns:
                df[c] = ""
        return df[COLS].copy()
    return pd.DataFrame(columns=COLS)


def _write(df: pd.DataFrame) -> None:
    df.to_csv(PROPS_PATH, index=False)


def _exists(df, d, h, a, model_player) -> bool:
    """Already logged for this fixture? Keyed on the matched MODEL player, so two book
    spellings of one player (e.g. 'Yuito Suzuki' / 'Y. Suzuki') can't double-log him."""
    if df.empty:
        return False
    return bool(((df["date"] == d) & (df["home_team"] == h)
                 & (df["away_team"] == a) & (df["model_player"] == model_player)).any())


def _freeze_event(df, event, home, away, neutral, model, now, lineup=None) -> list[dict]:
    best = props._best_book_prices(event)
    if not best:
        return []
    mtable, _ = props.goalscorer_table(home, away, neutral=neutral, model=model, lineup=lineup)
    if mtable.empty:
        return []
    names = mtable["player"].tolist()
    keys = {props.normkey(n): n for n in names}
    d = str(event.get("commence_time", ""))[:10] or str(date.today())
    # collapse book entries that map to the same model player to one row, keeping the
    # best (max) price -- otherwise differing book spellings double-log a player
    by_model: dict = {}
    for bk_name, info in best.items():
        mn = props.match_player(bk_name, names, keys)
        if mn is None:
            continue  # book player not in our top-5-league model -> can't price
        if mn not in by_model or info["price"] > by_model[mn][1]["price"]:
            by_model[mn] = (bk_name, info)
    rows = []
    for mn, (bk_name, info) in by_model.items():
        if _exists(df, d, home, away, mn):
            continue
        r = mtable[mtable["player"] == mn].iloc[0]
        book_p = 1.0 / info["price"]
        row = {c: "" for c in COLS}
        row.update({
            "logged_at": now, "model_fitted_on": model.fitted_on, "date": d,
            "home_team": home, "away_team": away, "neutral": neutral,
            "player": bk_name, "model_player": mn, "team": r["team"],
            "model_p": round(float(r["p_anytime"]), 4), "book_odds": info["price"],
            "book": info["book"], "book_p": round(book_p, 4),
            "edge": round(float(r["p_anytime"]) - book_p, 4), "result": "pending",
        })
        rows.append(row)
    return rows


# ---- API helpers (reuse odds.py plumbing) --------------------------------
def _odds():
    import odds
    odds.load_dotenv()
    return odds


def _pull_event(o, sport, event_id, regions):
    data, hdr = o.api_get(f"/sports/{sport}/events/{event_id}/odds/",
                          regions=regions, markets=GOAL_MARKET, oddsFormat="decimal")
    return data, hdr


def freeze_cmd(args) -> None:
    model = load_model()
    df = _read()
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    lineup = props.load_lineup(args.lineup) if args.lineup else None

    if args.event:                                   # offline: one saved event JSON
        event = json.load(open(args.event, encoding="utf-8"))
        new = _freeze_event(df, event, args.home, args.away, args.neutral, model, now, lineup)
    else:                                            # live: pull a date's slate
        o = _odds()
        sport = args.sport or o.discover_wc_key()
        if not sport:
            sys.exit("Could not auto-find the World Cup sport key; pass --sport.")
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ev_list, _ = o.api_get(f"/sports/{sport}/events/", dateFormat="iso",
                               commenceTimeFrom=now_iso)
        todays = [e for e in ev_list if str(e.get("commence_time", ""))[:10] == args.date]
        if not todays:
            dates = sorted({str(e.get("commence_time", ""))[:10] for e in ev_list})
            sys.exit(f"No events on {args.date}. Available: {dates}")
        neutral_by = o.neutral_lookup()
        cache: dict = {}
        hdr = {"remaining": "?", "used": "?"}
        new = []
        for e in todays:
            h = o.normalize(e["home_team"], model.teams, cache)
            a = o.normalize(e["away_team"], model.teams, cache)
            if h is None or a is None:
                print(f"  skip (unmapped teams): {e['home_team']} v {e['away_team']}")
                continue
            data, hdr = _pull_event(o, sport, e["id"], args.regions)
            neutral = neutral_by.get((h, a), True)
            rows = _freeze_event(df, data, h, a, neutral, model, now, lineup)
            print(f"  {h} v {a}: froze {len(rows)} player(s)")
            new += rows
        print(f"quota: {hdr['remaining']} remaining, {hdr['used']} used")

    if new:
        add = pd.DataFrame(new, columns=COLS)
        df = add if df.empty else pd.concat([df, add], ignore_index=True)
        _write(df)
    print(f"\nfroze {len(new)} player price(s); total tracked {len(df)} "
          f"({int((df['result'] == 'pending').sum())} pending).")


def close_cmd(args) -> None:
    """Re-pull current best prices into closing_odds for pending rows. Run near
    kickoff -- the last pre-kickoff snapshot is the true closing line. CLV uses the
    same best-price metric the bet was logged at (apples to apples)."""
    df = _read()
    pend = df[df["result"] == "pending"].copy()
    if args.date:
        pend = pend[pend["date"] == args.date]
    if pend.empty:
        sys.exit(f"No pending rows{' on ' + args.date if args.date else ''}.")
    o = _odds()
    sport = args.sport or o.discover_wc_key()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ev_list, _ = o.api_get(f"/sports/{sport}/events/", dateFormat="iso",
                           commenceTimeFrom=now_iso)
    cache: dict = {}
    ev_by_pair = {}
    for e in ev_list:
        h = o.normalize(e["home_team"], load_model().teams, cache)
        a = o.normalize(e["away_team"], load_model().teams, cache)
        if h is not None and a is not None:
            ev_by_pair[(h, a)] = e["id"]

    closed = missing = 0
    hdr = {"remaining": "?", "used": "?"}
    for (h, a), grp in pend.groupby(["home_team", "away_team"]):
        eid = ev_by_pair.get((h, a))
        if eid is None:
            missing += len(grp)
            continue
        data, hdr = _pull_event(o, sport, eid, args.regions)
        best = props._best_book_prices(data)
        for i in grp.index:
            price = best.get(df.at[i, "player"])
            if price is None:
                missing += 1
                continue
            df.at[i, "closing_odds"] = price["price"]
            closed += 1
    _write(df)
    print(f"closed {closed} price(s); {missing} not closeable (game started / player gone). "
          f"quota: {hdr['remaining']} remaining, {hdr['used']} used")


def grade_cmd(args) -> None:
    df = _read()
    if df.empty:
        sys.exit("No props tracked yet. Run `props_track.py freeze` first.")
    cutoff = getattr(args, "since", None) or WC_START
    gs = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "goalscorers.csv")
    gs = gs[(gs["own_goal"].astype(str).str.upper() != "TRUE")   # own goals don't count
            & (gs["date"].astype(str) >= cutoff)]                # tournament window only
    # normed scorer names per fixture, keyed on (home, away) -- robust to date diffs
    scorers: dict = {}
    for h, a, sc in zip(gs["home_team"], gs["away_team"], gs["scorer"]):
        scorers.setdefault((h, a), set()).add(props.normkey(sc))
    results = load_results()
    results = results[results["date"].astype(str) >= cutoff]
    played_pairs = set(zip(results.dropna(subset=["home_score"])["home_team"],
                           results.dropna(subset=["home_score"])["away_team"]))

    fill = ["scored", "result"]
    df[fill] = df[fill].astype(object)
    graded = 0
    for i, r in df.iterrows():
        if str(r["result"]) != "pending":
            continue
        key = (r["home_team"], r["away_team"])
        if key not in played_pairs:
            continue                              # game not in results data yet
        match_scorers = scorers.get(key, set())
        pk = props.normkey(r["player"])
        ln = pk.split()[-1] if pk else ""
        # exact normed name, or last-name match (book 'Lukaku' vs data 'R. Lukaku')
        hit = int(pk in match_scorers
                  or any(s.split() and s.split()[-1] == ln for s in match_scorers))
        df.at[i, "scored"] = hit
        df.at[i, "result"] = "graded"
        graded += 1
    _write(df)
    print(f"graded {graded}; {int((df['result'] == 'pending').sum())} still pending "
          f"(game not in results data yet).")


def report_cmd(args) -> None:
    df = _read()
    g = df[df["result"] == "graded"].copy()
    print(f"=== Anytime-goalscorer tracker ===  ({len(df)} tracked, {len(g)} graded)\n")
    if g.empty:
        print("Nothing graded yet -- run `props_track.py grade` after games are played.")
        return
    g["model_p"] = g["model_p"].astype(float)
    g["book_p"] = g["book_p"].astype(float)
    g["scored"] = g["scored"].astype(int)
    n = len(g)
    y = g["scored"].to_numpy()
    pm = g["model_p"].to_numpy()
    pb = g["book_p"].to_numpy()

    brier_m = float(np.mean((pm - y) ** 2))
    brier_b = float(np.mean((pb - y) ** 2))
    ll_m = float(-np.mean(y * np.log(np.clip(pm, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - pm, 1e-9, 1))))
    ll_b = float(-np.mean(y * np.log(np.clip(pb, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - pb, 1e-9, 1))))
    print(f"  base rate scored      {y.mean()*100:.1f}%  ({y.sum()}/{n})")
    print(f"  Brier   model {brier_m:.3f} | book {brier_b:.3f}  ({'model' if brier_m<brier_b else 'book'} better)")
    print(f"  log loss model {ll_m:.3f} | book {ll_b:.3f}  ({'model' if ll_m<ll_b else 'book'} better)")

    print("\n  calibration (model prob bucket -> actual scored rate):")
    edges = [0, .1, .2, .3, .4, .6, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        b = g[(pm >= lo) & (pm < hi)]
        if b.empty:
            continue
        print(f"    {lo*100:3.0f}-{hi*100:3.0f}%  n={len(b):3d}  "
              f"pred {b['model_p'].mean()*100:4.1f}%  actual {b['scored'].mean()*100:4.1f}%")

    # edge test: did model picks (model_p > book_p) win more than the price implied,
    # and would flat-staking them at the logged price have profited?
    val = g[g["edge"].astype(float) > 0]
    if not val.empty:
        roi = float((val["scored"] * val["book_odds"].astype(float) - 1).mean())
        print(f"\n  model edge picks (model_p > book_p): n={len(val)}  "
              f"hit {val['scored'].mean()*100:.1f}%  vs book-implied "
              f"{val['book_p'].mean()*100:.1f}%  |  flat-stake ROI {roi*100:+.1f}%")
    clv = g[g["closing_odds"].astype(str).str.strip().replace("nan", "") != ""]
    if not clv.empty:
        bo = clv["book_odds"].astype(float)
        co = clv["closing_odds"].astype(float)
        beat = float((bo > co).mean())   # our price longer than close = +CLV
        print(f"  CLV: {len(clv)} closed | beat the close {beat*100:.0f}% "
              f"(logged avg {bo.mean():.2f} vs close {co.mean():.2f})")


def main() -> None:
    p = argparse.ArgumentParser(description="Anytime-goalscorer forward tracker")
    sub = p.add_subparsers(dest="cmd", required=True)

    fz = sub.add_parser("freeze", help="lock model probs + book prices for a slate")
    fz.add_argument("--date", default=str(date.today() + timedelta(days=1)),
                    help="match date YYYY-MM-DD to pull live (default tomorrow)")
    fz.add_argument("--event", help="offline: a saved event-odds JSON instead of pulling")
    fz.add_argument("home", nargs="?", help="home team (offline --event mode)")
    fz.add_argument("away", nargs="?", help="away team (offline --event mode)")
    fz.add_argument("--neutral", action="store_true")
    fz.add_argument("--regions", default=PROPS_REGION)
    fz.add_argument("--sport")
    fz.add_argument("--lineup", help="lineup JSON to set minutes from the named XI")
    fz.set_defaults(func=freeze_cmd)

    cl = sub.add_parser("close", help="snapshot closing prices for pending rows (near kickoff)")
    cl.add_argument("--date", default=str(date.today()))
    cl.add_argument("--regions", default=PROPS_REGION)
    cl.add_argument("--sport")
    cl.set_defaults(func=close_cmd)

    gr = sub.add_parser("grade", help="fill who scored from goalscorers.csv")
    gr.add_argument("--since", default=WC_START, help="only grade games on/after this date")
    gr.set_defaults(func=grade_cmd)
    sub.add_parser("report", help="calibration + edge scoreboard").set_defaults(func=report_cmd)

    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.func(args)


if __name__ == "__main__":
    main()
