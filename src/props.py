"""World Cup anytime-goalscorer model.

A player's probability of scoring in a given WC match is driven by three things:
how many shots they take (volume), how good those shots are (shot quality), and
how many goals their team is expected to score (match context).

  * Volume + current role come from this season's club data (FBref top-5 leagues,
    2025/26, ``data/club_players_2025_2026.csv``): shots per 90 and expected minutes.
  * Shot quality comes from Understat expected-goals history (xG per shot over the
    last three seasons, ``data/understat_players.csv``) -- far more stable than a
    single season's goals/shot, and the fix for elite finishers being underrated
    by raw conversion.
  * Match context is the Dixon-Coles team lambda for the specific fixture, which
    scales the player's baseline up against weak defences and down against strong.

      lambda_player = shots90 * xg_per_shot * (exp_min/90) * opp_scale * INTL_FACTOR
      P(scores >= 1) = 1 - exp(-lambda_player)            # Poisson

Honest limits: top-5-league players only (no MLS / Saudi / Eredivisie / etc.);
expected minutes are a club-form estimate, NOT lineup news (the single biggest
error source -- see props_track.py for forward validation); and the
club->international level haircut (INTL_FACTOR) is an unvalidated global knob.
There is no free historical prop-odds data to fit it against, so the only honest
test of this model is forward-tracking, not a backtest.

    python src/props.py table Ecuador Germany           # model goalscorer board
    python src/props.py compare event.json Ecuador Germany   # vs saved book odds
    python src/props.py template Germany Ecuador > xi.json   # editable lineup skeleton
    python src/props.py table Germany Ecuador --lineup xi.json   # minutes from named XI
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from predict import load_model

DATA = Path(__file__).resolve().parent.parent / "data"
CLUB_PATH = DATA / "club_players_2025_2026.csv"
XG_PATH = DATA / "understat_players.csv"

LEAGUE_XGPS = 0.10      # league-average expected goals per shot (shrinkage target)
LEAGUE_CONV = 0.105     # league-average goals per shot (goal-based fallback)
SHRINK_K = 25.0         # shots of league-average pull on the conversion estimate
SEASON_DECAY = 0.7      # weight per season back, for the xG history blend
MIN_CLUB_MINUTES = 270  # >= 3 full club matches before a rate is trusted
TEAM_BASELINE = 1.35    # typical team expected goals/match (opp_scale anchor)
INTL_FACTOR = 0.90      # club -> international level haircut (UNVALIDATED knob)
STARTER_MIN = 82.0      # expected minutes for a confirmed starter (lineup override)
SUB_MIN = 23.0          # expected minutes for a named substitute (lineup override)

# FBref 3-letter nation code -> DC / Odds-API country name (WC-relevant nations).
CODE = {
    "ECU": "Ecuador", "GER": "Germany", "FRA": "France", "ESP": "Spain",
    "ITA": "Italy", "ENG": "England", "BRA": "Brazil", "NED": "Netherlands",
    "ARG": "Argentina", "POR": "Portugal", "BEL": "Belgium", "CRO": "Croatia",
    "USA": "United States", "MEX": "Mexico", "URU": "Uruguay", "COL": "Colombia",
    "JPN": "Japan", "KOR": "South Korea", "SEN": "Senegal", "MAR": "Morocco",
    "CIV": "Ivory Coast", "GHA": "Ghana", "NGA": "Nigeria", "CMR": "Cameroon",
    "SUI": "Switzerland", "DEN": "Denmark", "POL": "Poland", "SRB": "Serbia",
    "AUT": "Austria", "SWE": "Sweden", "NOR": "Norway", "TUR": "Turkey",
    "WAL": "Wales", "SCO": "Scotland", "UKR": "Ukraine", "CZE": "Czech Republic",
    "TUN": "Tunisia", "ALG": "Algeria", "EGY": "Egypt", "AUS": "Australia",
    "IRN": "Iran", "KSA": "Saudi Arabia", "QAT": "Qatar", "PAR": "Paraguay",
    "PER": "Peru", "CHI": "Chile", "VEN": "Venezuela", "CRC": "Costa Rica",
    "PAN": "Panama", "GRE": "Greece", "ROU": "Romania", "HUN": "Hungary",
    "SVK": "Slovakia", "SVN": "Slovenia", "ALB": "Albania", "GEO": "Georgia",
    "ISR": "Israel", "IRQ": "Iraq", "JOR": "Jordan", "UZB": "Uzbekistan",
    "CPV": "Cape Verde", "RSA": "South Africa", "MLI": "Mali", "BFA": "Burkina Faso",
    "NIR": "Northern Ireland", "IRL": "Ireland", "ARM": "Armenia", "CUW": "Curacao",
}


# Latin letters that do NOT decompose under NFKD (so accent-stripping misses them);
# folded explicitly so 'Groß'/'Gross', 'Højbjerg'/'Hojbjerg' etc. match across sources.
_FOLD = str.maketrans({"ß": "ss", "ø": "o", "Ø": "o", "ł": "l", "Ł": "l",
                       "đ": "d", "Đ": "d", "ð": "d", "þ": "th", "æ": "ae", "œ": "oe"})


def normkey(name: str) -> str:
    """Accent- and case-insensitive key for matching names across data sources
    (FBref 'Kylian Mbappe' vs Understat 'Kylian Mbappé' vs a book's spelling)."""
    n = unicodedata.normalize("NFKD", str(name).translate(_FOLD))
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(n.lower().replace(".", " ").split())


@lru_cache(maxsize=1)
def _xg_lookup() -> dict:
    """normkey(player) -> expected goals per shot, from Understat history.

    Recent seasons and higher-minute seasons count more; the per-player rate is
    shrunk toward the league average so a small-sample season can't dominate.
    Total xG / total shots (penalties included) matches the anytime-scorer market,
    where penalties count toward the player's goal."""
    if not XG_PATH.exists():
        return {}
    us = pd.read_csv(XG_PATH)
    w = SEASON_DECAY ** (us["year"].max() - us["year"])
    us = us.assign(wx=us["xG"] * w, ws=us["shots"] * w)
    us["key"] = us["player_name"].map(normkey)
    g = us.groupby("key").agg(xG=("wx", "sum"), shots=("ws", "sum")).reset_index()
    g = g[g["shots"] > 0]
    xgps = (g["xG"] + SHRINK_K * LEAGUE_XGPS) / (g["shots"] + SHRINK_K)
    return dict(zip(g["key"], xgps))


@lru_cache(maxsize=1)
def build_profiles() -> pd.DataFrame:
    """One row per WC-eligible top-5-league player: shot volume, conversion, and an
    expected-minutes estimate. Conversion is xG-per-shot when the player has
    Understat history, else a shrunk goals-per-shot fallback."""
    club = pd.read_csv(CLUB_PATH)
    club["natcode"] = club["Nation"].astype(str).str.split().str[-1]
    g = (club.groupby(["Player", "natcode"])
         .agg(Min=("Min", "sum"), Sh=("Sh", "sum"), SoT=("SoT", "sum"),
              Gls=("Gls", "sum"), MP=("MP", "sum"), Starts=("Starts", "sum"),
              Pos=("Pos", "first"))
         .reset_index())
    g = g[(g["Min"] >= MIN_CLUB_MINUTES) & (g["Sh"] > 0)].copy()
    g["country"] = g["natcode"].map(CODE)
    g = g[g["country"].notna()]
    g["shots90"] = g["Sh"] / (g["Min"] / 90.0)

    xg = _xg_lookup()
    key = g["Player"].map(normkey)
    g["xgps"] = key.map(xg)
    goal_conv = (g["Gls"] + SHRINK_K * LEAGUE_CONV) / (g["Sh"] + SHRINK_K)
    g["conv"] = g["xgps"].fillna(goal_conv)
    g["conv_src"] = np.where(g["xgps"].notna(), "xg", "goals")

    # expected minutes: a regular starter is assumed to go ~a full match; a rotation
    # player gets their club minutes-per-appearance. Not lineup news -- a prior.
    min_per_start = np.where(g["Starts"] > 0, g["Min"] / g["Starts"], g["Min"] / g["MP"])
    starter = g["Starts"] >= g["MP"] * 0.5
    g["exp_min"] = np.where(starter, np.clip(min_per_start, 60.0, 90.0),
                            np.clip(min_per_start, 0.0, 90.0))
    return g.reset_index(drop=True)


def resolve_lineup(team_lu: dict, model_names: list[str]):
    """Map a fixture's named XI to model players. Returns (roles, unmatched): roles is
    {model_player: 'starter'|'sub'}; unmatched is the lineup names with no top-5-league
    profile -- the coverage gaps, i.e. real starters the model cannot price."""
    keys = {normkey(n): n for n in model_names}
    roles, unmatched = {}, []
    for role, names in [("starter", team_lu.get("starters", [])),
                        ("sub", team_lu.get("bench", []))]:
        for nm in names:
            mp = match_player(nm, model_names, keys)
            if mp is None:
                unmatched.append(nm)
            else:
                roles[mp] = role
    return roles, unmatched


def goalscorer_table(home: str, away: str, neutral: bool = False, model=None,
                     lineup: dict | None = None):
    """DataFrame of P(anytime goalscorer) for both squads' top-5-league players, plus
    the (lambda_home, lambda_away) the match was priced at.

    With `lineup` ({team: {'starters':[...], 'bench':[...]}}), expected minutes come
    from the named XI -- confirmed starters get STARTER_MIN, named subs SUB_MIN, and
    players not in the XI are dropped (not playing). Without it, minutes fall back to
    the club-form prior. The lineup is the single biggest accuracy lever and the only
    one that lets the model react to team news faster than soft books reprice."""
    m = model or load_model()
    lam_home, lam_away = m.lambdas(home, away, neutral=neutral)
    prof = build_profiles()
    out = []
    for team, lam in [(home, lam_home), (away, lam_away)]:
        sub = prof[prof["country"] == team]
        if sub.empty:
            continue
        opp_scale = float(np.clip(lam / TEAM_BASELINE, 0.45, 2.0))
        team_lu = (lineup or {}).get(team)
        roles = resolve_lineup(team_lu, sub["Player"].tolist())[0] if team_lu else None
        for _, r in sub.iterrows():
            if roles is not None:
                role = roles.get(r["Player"])
                if role is None:
                    continue                    # named XI given, player not in it
                exp_min = STARTER_MIN if role == "starter" else SUB_MIN
            else:
                exp_min = r["exp_min"]
            lam_p = (r["shots90"] * r["conv"] * (exp_min / 90.0)
                     * opp_scale * INTL_FACTOR)
            p_any = 1.0 - np.exp(-lam_p)
            out.append({"team": team, "player": r["Player"], "p_anytime": p_any,
                        "fair_odds": (1.0 / p_any) if p_any > 0 else np.inf,
                        "shots90": r["shots90"], "conv": r["conv"],
                        "conv_src": r["conv_src"], "exp_min": exp_min,
                        "MP": int(r["MP"])})
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("p_anytime", ascending=False).reset_index(drop=True)
    return df, (lam_home, lam_away)


def load_lineup(path: str) -> dict:
    """Read a lineup JSON: {team: {"starters": [names], "bench": [names]}}. Names are
    matched to model players fuzzily, so last names / alternate spellings are fine."""
    raw = json.load(open(path, encoding="utf-8"))
    return {team: {"starters": v.get("starters", []), "bench": v.get("bench", [])}
            for team, v in raw.items()}


def match_player(book_name: str, candidates: list[str], keys: dict | None = None):
    """Best model-player match for a book's player name. Exact accent-folded match
    first, then a last-name containment check, then fuzzy. None if nothing close."""
    keys = keys or {normkey(c): c for c in candidates}
    bk = normkey(book_name)
    if bk in keys:
        return keys[bk]
    ln = bk.split()[-1] if bk else ""
    hits = [c for c in candidates if normkey(c).split() and normkey(c).split()[-1] == ln]
    if len(hits) == 1:
        return hits[0]
    fuzzy = difflib.get_close_matches(bk, list(keys), n=1, cutoff=0.85)
    return keys[fuzzy[0]] if fuzzy else None


# ---- CLI -----------------------------------------------------------------
def _lineup_note(lineup, home, away) -> None:
    """Warn about confirmed XI players the model can't price (non-top-5 coverage gaps)."""
    if not lineup:
        return
    for team in (home, away):
        team_lu = lineup.get(team)
        if not team_lu:
            continue
        names = build_profiles()
        names = names[names["country"] == team]["Player"].tolist()
        _, unmatched = resolve_lineup(team_lu, names)
        if unmatched:
            print(f"  ! {team}: no model price for {', '.join(unmatched)} "
                  f"(not top-5-league -- coverage gap)")


def _table_cmd(args) -> None:
    lineup = load_lineup(args.lineup) if args.lineup else None
    df, (lh, la) = goalscorer_table(args.home, args.away, neutral=args.neutral, lineup=lineup)
    tag = "  [lineup applied]" if lineup else ""
    print(f"\n{args.home} ({lh:.2f}) vs {args.away} ({la:.2f})  "
          f"[neutral={args.neutral}]  -- anytime goalscorer{tag}\n")
    _lineup_note(lineup, args.home, args.away)
    if df.empty:
        print("No top-5-league players found for these teams.")
        return
    print(f"  {'team':13} {'player':24} {'P':>5}  {'fair':>5}  detail")
    for _, r in df.head(args.n).iterrows():
        tag = "xg" if r["conv_src"] == "xg" else "g!"   # g! = goal-based fallback
        print(f"  {r['team']:13} {r['player'][:23]:24} {r['p_anytime']*100:4.1f}% "
              f"{r['fair_odds']:5.2f}  ({r['shots90']:.2f} sh/90, conv {r['conv']:.3f} "
              f"[{tag}], {r['exp_min']:.0f}min, {r['MP']}MP)")
    n_xg = int((df["conv_src"] == "xg").sum())
    print(f"\n{len(df)} players priced; {n_xg} on xG shot-quality, "
          f"{len(df) - n_xg} on goal-based fallback (no Understat history).")


def _best_book_prices(event: dict) -> dict:
    """player name -> {'price','book'} at the best (max) anytime-scorer price."""
    best: dict = {}
    for b in event.get("bookmakers", []):
        for m in b.get("markets", []):
            if m.get("key") != "player_goal_scorer_anytime":
                continue
            for o in m.get("outcomes", []):
                nm = o.get("description")
                if nm and (nm not in best or o["price"] > best[nm]["price"]):
                    best[nm] = {"price": o["price"], "book": b["title"]}
    return best


def _compare_cmd(args) -> None:
    event = json.load(open(args.event, encoding="utf-8"))
    best = _best_book_prices(event)
    lineup = load_lineup(args.lineup) if args.lineup else None
    model, (lh, la) = goalscorer_table(args.home, args.away, neutral=args.neutral, lineup=lineup)
    if model.empty:
        sys.exit("Model has no players for these teams (not top-5-league).")
    names = model["player"].tolist()
    keys = {normkey(n): n for n in names}
    rows, missing = [], []
    for bk_name, info in best.items():
        mn = match_player(bk_name, names, keys)
        if mn is None:
            missing.append(bk_name)
            continue
        r = model[model["player"] == mn].iloc[0]
        book_p = 1.0 / info["price"]
        rows.append({"player": bk_name, "team": r["team"], "model_p": r["p_anytime"],
                     "book_odds": info["price"], "book": info["book"], "book_p": book_p,
                     "edge": r["p_anytime"] - book_p})
    df = pd.DataFrame(rows).sort_values("edge", ascending=False)
    print(f"\n{args.home} ({lh:.2f}) vs {args.away} ({la:.2f})  -- anytime goalscorer: "
          f"model vs best book price\n")
    print(f"  {'player':24} {'team':12} {'model':>6} {'book':>6} {'odds':>6} {'edge':>7}  book")
    for _, r in df.iterrows():
        print(f"  {r['player'][:23]:24} {r['team'][:11]:12} {r['model_p']*100:5.1f}% "
              f"{r['book_p']*100:5.1f}% {r['book_odds']:6.2f} {r['edge']*100:+6.1f}pp  {r['book']}")
    print(f"\nCoverage: {len(df)}/{len(best)} priced players modeled."
          + (f"  No model price for: {', '.join(missing[:10])}" if missing else ""))
    print("NOTE: visible edges are confounded (minutes assumptions, no lineup news) "
          "until\nforward-tracking proves them -- see `props_track.py`.")


def _template_cmd(args) -> None:
    """Print an editable lineup JSON seeded with each side's most-likely scorers, so
    you only have to move names between 'starters' and 'bench' and add any non-top-5
    starters the model doesn't know. Pipe to a file, edit, then pass via --lineup."""
    df, _ = goalscorer_table(args.home, args.away, neutral=args.neutral)
    tmpl = {}
    for team in (args.home, args.away):
        names = df[df["team"] == team]["player"].tolist()
        tmpl[team] = {"starters": names[:11], "bench": names[11:18]}
    print("// seeded by shot rate (attackers first); reorder to the real XI",
          file=sys.stderr)
    print(json.dumps(tmpl, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser(description="World Cup anytime-goalscorer model")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("table", help="model goalscorer board for one fixture")
    t.add_argument("home"); t.add_argument("away")
    t.add_argument("--neutral", action="store_true")
    t.add_argument("-n", type=int, default=20, help="rows to show (default 20)")
    t.add_argument("--lineup", help="lineup JSON to set minutes from the named XI")
    t.set_defaults(func=_table_cmd)

    c = sub.add_parser("compare", help="model vs best book price from a saved event JSON")
    c.add_argument("event", help="Odds-API event-odds JSON with player_goal_scorer_anytime")
    c.add_argument("home"); c.add_argument("away")
    c.add_argument("--neutral", action="store_true")
    c.add_argument("--lineup", help="lineup JSON to set minutes from the named XI")
    c.set_defaults(func=_compare_cmd)

    tm = sub.add_parser("template", help="print an editable lineup JSON skeleton")
    tm.add_argument("home"); tm.add_argument("away")
    tm.add_argument("--neutral", action="store_true")
    tm.set_defaults(func=_template_cmd)

    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.func(args)


if __name__ == "__main__":
    main()
