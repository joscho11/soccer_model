"""Automated odds via The Odds API (free tier) -> value scan + auto-logging.

Replaces the manual `slate.py log` step: pulls live odds from US books
(FanDuel, DraftKings, BetMGM, ...) through one sanctioned JSON endpoint, matches
each game to the model's fixtures, takes the *best available price* per market
across books, and compares it to the calibrated model probability to surface
+EV bets. No scraping, no ToS problems, no extra pip deps (stdlib only).

Setup: get a free key at https://the-odds-api.com (free tier ~500 credits/month;
cost = markets x regions, so a h2h+totals/us scan is 2 credits). Then:

    export ODDS_API_KEY=...           # PowerShell: $env:ODDS_API_KEY="..."
    python src/odds.py sports          # find the World Cup sport key
    python src/odds.py scan            # value table for upcoming games
    python src/odds.py log --min-edge 0.02   # auto-log +EV bets to data/bet_log.csv

Team names differ between books and the dataset ("Czechia" vs "Czech Republic",
"USA" vs "United States"); an alias map plus fuzzy fallback handles it and any
unmatched game is reported rather than silently dropped.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import numpy as np

from dixon_coles import load_results
from predict import load_model
import markets
import slate

API_BASE = "https://api.the-odds-api.com/v4"

# Odds-API team name -> martj42 dataset name. Extend as mismatches surface.
ALIASES = {
    "USA": "United States", "United States of America": "United States",
    "Czechia": "Czech Republic", "South Korea": "South Korea",
    "Korea Republic": "South Korea", "IR Iran": "Iran", "Iran": "Iran",
    "Ivory Coast": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast", "Cape Verde": "Cape Verde",
    "Cabo Verde": "Cape Verde", "DR Congo": "DR Congo",
    "Congo DR": "DR Congo", "Curacao": "Curaçao", "Türkiye": "Turkey",
    "Turkiye": "Turkey", "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


# ---- API plumbing --------------------------------------------------------
def load_dotenv() -> None:
    """Load KEY=VALUE lines from a .env (project root or cwd) into the environment.
    No dependency; real exported env vars take precedence (setdefault)."""
    from pathlib import Path
    for base in (Path(__file__).resolve().parent.parent, Path.cwd()):
        envp = base / ".env"
        if not envp.exists():
            continue
        for line in envp.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return


def api_key() -> str:
    load_dotenv()
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("ODDS_API_KEY not set. Get a free key at https://the-odds-api.com "
                 "then `export ODDS_API_KEY=...` (PowerShell: $env:ODDS_API_KEY=\"...\").")
    return key


def api_get(path: str, **params) -> tuple[object, dict]:
    params["apiKey"] = api_key()
    url = f"{API_BASE}{path}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=30) as r:
            data = json.loads(r.read())
            hdr = {"remaining": r.headers.get("x-requests-remaining"),
                   "used": r.headers.get("x-requests-used")}
            return data, hdr
    except HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        if e.code == 401:
            sys.exit(f"401 Unauthorized — bad or missing API key. ({body})")
        if e.code == 429:
            sys.exit(f"429 — out of quota for the month. ({body})")
        sys.exit(f"HTTP {e.code}: {body}")
    except URLError as e:
        sys.exit(f"Network error reaching The Odds API: {e.reason}")


def american(dec: float) -> str:
    """Decimal -> American odds string (e.g. 2.50 -> '+150', 1.50 -> '-200')."""
    if dec >= 2.0:
        return f"+{round((dec - 1) * 100)}"
    return f"-{round(100 / (dec - 1))}"


def discover_wc_key() -> str | None:
    sports, _ = api_get("/sports/", all="true")
    EXCLUDE = ("women", "winner", "club", "qualifier")  # outrights / wrong comp
    cands = [s for s in sports if s.get("group") == "Soccer"
             and "world cup" in (s.get("title", "") + s.get("key", "")).lower()
             and not any(x in (s.get("title", "") + s.get("key", "")).lower()
                         for x in EXCLUDE)]
    if not cands:
        return None
    # exact match wins, then prefer active (in-season)
    exact = [s for s in cands if s["key"] == "soccer_fifa_world_cup"]
    if exact:
        return exact[0]["key"]
    cands.sort(key=lambda s: (not s.get("active", False)))
    return cands[0]["key"]


# ---- venue ---------------------------------------------------------------
def neutral_lookup() -> dict:
    """(home, away) dataset names -> neutral flag, from unplayed WC fixtures. Host
    nations (USA/Canada/Mexico) play at home, so not every WC game is neutral."""
    df = load_results()
    df = df[(df["tournament"] == "FIFA World Cup") & df["home_score"].isna()]
    return {(r["home_team"], r["away_team"]):
            str(r["neutral"]).upper() == "TRUE" for _, r in df.iterrows()}


# ---- name matching -------------------------------------------------------
def normalize(name: str, teams: list[str], cache: dict) -> str | None:
    if name in cache:
        return cache[name]
    cand = ALIASES.get(name, name)
    if cand in teams:
        cache[name] = cand
        return cand
    close = difflib.get_close_matches(cand, teams, n=1, cutoff=0.85)
    cache[name] = close[0] if close else None
    return cache[name]


# ---- value extraction ----------------------------------------------------
def collect(event: dict) -> dict:
    """selection key -> list of (price, book) across all books."""
    d: dict = {}
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for oc in mk.get("outcomes", []):
                if mk["key"] == "h2h":
                    key = ("h2h", oc["name"])
                elif mk["key"] == "totals" and oc.get("point") == 2.5:
                    key = ("tot25", oc["name"])  # name is "Over"/"Under"
                elif mk["key"] == "btts":
                    key = ("btts", oc["name"])   # name is "Yes"/"No"
                else:
                    continue
                d.setdefault(key, []).append((oc["price"], bk["key"]))
    return d


def novig(keys, collected) -> dict | None:
    """No-vig consensus probability per outcome: average implied prob across books,
    then normalise so the mutually-exclusive set sums to 1 (removes the bookmaker
    margin). This is the *efficient-market* benchmark — far more honest than the
    single best line, which just reflects whichever soft book hung a loose number."""
    imp = {}
    for k in keys:
        if k not in collected:
            return None
        imp[k] = float(np.mean([1.0 / p for p, _ in collected[k]]))
    s = sum(imp.values())
    return {k: imp[k] / s for k in keys}


def scan_event(model, event, name_cache, neutral_by_fixture, min_books: int = 3):
    """Return (matched, rows). Edge is measured vs the de-vigged market consensus
    (genuine disagreement), with EV computed at the best available price."""
    api_h, api_a = event["home_team"], event["away_team"]
    h = normalize(api_h, model.teams, name_cache)
    a = normalize(api_a, model.teams, name_cache)
    if h is None or a is None:
        return False, (api_h if h is None else api_a)

    mdate = str(event.get("commence_time", ""))[:10]  # real kickoff date from the API
    neutral = neutral_by_fixture.get((h, a), True)  # host nations play at home
    mat = model.score_matrix(h, a, neutral=neutral)
    o = markets.outcome_probs(mat)
    over = model.over_prob(h, a, neutral=neutral, line=2.5, calibrated=True)
    bt = markets.btts(mat)
    coll = collect(event)
    h2h_nv = novig([("h2h", api_h), ("h2h", "Draw"), ("h2h", api_a)], coll)
    tot_nv = novig([("tot25", "Over"), ("tot25", "Under")], coll)
    btts_nv = novig([("btts", "Yes"), ("btts", "No")], coll)

    # (label, market_code, line, model_prob, price_key, novig_dict)
    sels = [
        ("home", "home", "", o["home"], ("h2h", api_h), h2h_nv),
        ("draw", "draw", "", o["draw"], ("h2h", "Draw"), h2h_nv),
        ("away", "away", "", o["away"], ("h2h", api_a), h2h_nv),
        ("over 2.5", "over", 2.5, over, ("tot25", "Over"), tot_nv),
        ("under 2.5", "under", 2.5, 1 - over, ("tot25", "Under"), tot_nv),
        ("btts yes", "btts_yes", "", bt["btts_yes"], ("btts", "Yes"), btts_nv),
        ("btts no", "btts_no", "", bt["btts_no"], ("btts", "No"), btts_nv),
    ]
    rows = []
    for label, code, line, prob, pkey, nv in sels:
        if pkey not in coll or nv is None or len(coll[pkey]) < min_books:
            continue
        price, book = max(coll[pkey], key=lambda pb: pb[0])  # best available price
        mkt = nv[pkey]                                       # no-vig market prob
        disagree = prob - mkt                                # genuine model edge
        v = markets.value_vs_odds(prob, price)
        # value = model disagrees with the efficient market AND a real price clears it
        value = disagree > 0 and v["ev_per_unit"] > 0
        rows.append({"label": label, "market": code, "line": line, "model_prob": prob,
                     "mkt_prob": mkt, "disagree": disagree, "price": price, "book": book,
                     "ev": v["ev_per_unit"], "value": value, "home": h, "away": a,
                     "date": mdate})
    return True, rows


# ---- commands ------------------------------------------------------------
def sports_cmd(args):
    sports, hdr = api_get("/sports/", all="true")
    soccer = [s for s in sports if s.get("group") == "Soccer"]
    print(f"{len(soccer)} soccer competitions (active marked *):\n")
    for s in sorted(soccer, key=lambda s: (not s.get("active", False), s["key"])):
        star = "*" if s.get("active") else " "
        print(f"  {star} {s['key']:38s} {s['title']}")
    print(f"\nquota: {hdr['remaining']} remaining, {hdr['used']} used this month")


def _gather(args):
    model = load_model()
    sport = args.sport or discover_wc_key()
    if not sport:
        sys.exit("Could not auto-find a World Cup sport key. Run `odds.py sports` "
                 "and pass --sport <key>.")
    # pre-match only: skip games already started (in-play / locked prices are noise)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # h2h + totals are on the cheap bulk endpoint; BTTS/props/alt-totals are
    # event-level only (1 credit per game) so we don't pull them in a broad scan.
    events, hdr = api_get(f"/sports/{sport}/odds/", regions=args.regions,
                          markets="h2h,totals", oddsFormat="decimal",
                          commenceTimeFrom=now_iso)
    name_cache: dict = {}
    neutral_by_fixture = neutral_lookup()
    matched, unmatched = [], []
    for ev in events:
        ok, payload = scan_event(model, ev, name_cache, neutral_by_fixture)
        (matched if ok else unmatched).append(payload)
    return model, sport, events, matched, unmatched, hdr


def scan_cmd(args):
    _, sport, events, matched, unmatched, hdr = _gather(args)
    print(f"sport={sport}  region={args.regions}  events={len(events)}  "
          f"value bets, decimal odds >= {args.min_odds} ({american(args.min_odds)}), "
          f"edge >= {args.min_edge*100:.0f}pp\n")
    # flatten to value bets meeting the upside/odds preference, best edge first
    picks = [r for rows in matched for r in rows
             if r["value"] and r["disagree"] >= args.min_edge and r["price"] >= args.min_odds
             and (not args.date or r["date"] == args.date)]
    picks.sort(key=lambda r: (-r["ev"], -r["disagree"]))
    for r in picks:
        plus = "  <-- +odds upside" if r["price"] >= 2.0 else ""
        print(f"{american(r['price']):>5} ({r['price']:>5})  {r['home']} v {r['away']}"
              f"  [{r['label']}]  model {r['model_prob']*100:.0f}% vs mkt "
              f"{r['mkt_prob']*100:.0f}%  edge {r['disagree']*100:+.1f}pp  "
              f"ev {r['ev']:+.2f}  ({r['book']}){plus}")
    if not picks:
        print(f"No value bets at decimal >= {args.min_odds}. Lower --min-odds/--min-edge.")
    if unmatched:
        print(f"\nunmatched teams (add to ALIASES): {sorted(set(unmatched))}")
    print(f"\nquota: {hdr['remaining']} remaining, {hdr['used']} used")


def btts_cmd(args):
    """Both-teams-to-score value, via the per-event endpoint (1 credit per game)."""
    model = load_model()
    sport = args.sport or discover_wc_key()
    if not sport:
        sys.exit("Could not auto-find a World Cup sport key; pass --sport.")
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ev_list, hdr = api_get(f"/sports/{sport}/events/", dateFormat="iso",
                           commenceTimeFrom=now_iso)  # listing events is free
    todays = [e for e in ev_list if str(e.get("commence_time", ""))[:10] == args.date]
    if not todays:
        dates = sorted({str(e.get("commence_time", ""))[:10] for e in ev_list})
        print(f"No events on {args.date}. Available dates: {dates}")
        return
    nl = neutral_lookup()
    cache: dict = {}
    print(f"BTTS value on {args.date}, decimal odds >= {args.min_odds} "
          f"({american(args.min_odds)})  [{len(todays)} games, 1 credit each]\n")
    shown = 0
    for e in todays:
        h = normalize(e["home_team"], model.teams, cache)
        a = normalize(e["away_team"], model.teams, cache)
        if h is None or a is None:
            continue
        data, hdr = api_get(f"/sports/{sport}/events/{e['id']}/odds/",
                            regions=args.regions, markets="btts", oddsFormat="decimal")
        coll = collect(data)
        nv = novig([("btts", "Yes"), ("btts", "No")], coll)
        if nv is None:
            continue  # no BTTS market offered for this game
        neu = nl.get((h, a), True)
        b = markets.btts(model.score_matrix(h, a, neutral=neu))
        for name, mp in [("Yes", b["btts_yes"]), ("No", b["btts_no"])]:
            key = ("btts", name)
            if key not in coll:
                continue
            price, book = max(coll[key], key=lambda pb: pb[0])
            disagree = mp - nv[key]
            v = markets.value_vs_odds(mp, price)
            if v["value"] and disagree > 0 and price >= args.min_odds:
                print(f"{american(price):>5} ({price:>5})  {h} v {a} [BTTS {name}]  "
                      f"model {mp*100:.0f}% vs mkt {nv[key]*100:.0f}%  "
                      f"edge {disagree*100:+.1f}pp  ev {v['ev_per_unit']:+.2f}  ({book})")
                shown += 1
    if not shown:
        print("No BTTS value clears your odds floor on this date.")
    print(f"\nquota: {hdr['remaining']} remaining, {hdr['used']} used")


def log_cmd(args):
    _, sport, events, matched, unmatched, hdr = _gather(args)
    logged = skipped = 0
    for rows in matched:
        for r in rows:
            if not r["value"] or r["disagree"] < args.min_edge:
                continue
            if slate.pending_exists(r["home"], r["away"], r["market"], r["line"]):
                skipped += 1
                continue
            slate.record_bet(r["home"], r["away"], r["market"], r["price"],
                             r["model_prob"], r["date"], line=(r["line"] or 2.5),
                             stake=args.stake)
            logged += 1
            print(f"  logged {r['home']} v {r['away']} [{r['label']}] @ {r['price']} "
                  f"({r['book']}) edge {r['disagree']*100:+.1f}pp vs mkt")
    print(f"\nauto-logged {logged} +EV bet(s) (edge >= {args.min_edge}); "
          f"{skipped} already pending.")
    if unmatched:
        print(f"unmatched teams (add to ALIASES): {sorted(set(unmatched))}")
    print(f"quota: {hdr['remaining']} remaining, {hdr['used']} used")


def main():
    p = argparse.ArgumentParser(description="The Odds API value scan + auto-logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sports", help="list soccer sport keys + quota")
    sp.set_defaults(func=sports_cmd)

    sc = sub.add_parser("scan", help="value table for upcoming games")
    sc.add_argument("--sport", help="sport key (default: auto-find World Cup)")
    sc.add_argument("--regions", default="us", help="us, uk, eu, au (comma-sep)")
    sc.add_argument("--min-edge", type=float, default=0.03,
                    help="min model-vs-market disagreement to show (prob, default 0.03=3pp)")
    sc.add_argument("--min-odds", type=float, default=1.67,
                    help="min decimal odds (default 1.67 = -150; upside preference)")
    sc.add_argument("--date", default=None, help="restrict to one match date YYYY-MM-DD")
    sc.set_defaults(func=scan_cmd)

    bt = sub.add_parser("btts", help="both-teams-to-score value (per-event, 1 credit/game)")
    bt.add_argument("--date", default=str(date.today() + timedelta(days=1)),
                    help="match date YYYY-MM-DD (default tomorrow)")
    bt.add_argument("--regions", default="us")
    bt.add_argument("--sport", default=None)
    bt.add_argument("--min-odds", type=float, default=1.67)
    bt.set_defaults(func=btts_cmd)

    lg = sub.add_parser("log", help="auto-log +EV bets to the bet log")
    lg.add_argument("--sport")
    lg.add_argument("--regions", default="us")
    lg.add_argument("--min-edge", type=float, default=0.03,
                    help="min model-vs-market disagreement to log (prob, default 0.03=3pp)")
    lg.add_argument("--stake", type=float, default=1.0)
    lg.set_defaults(func=log_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
