"""Forward prediction tracker: freeze point-in-time model predictions, grade them
against actual results, and report accuracy — so we can honestly compare what the
model called to what happened.

The honest test of a predictive model is how its *pre-game* forecasts fare, so a
prediction is locked when first frozen (by a model fit before the match) and never
rewritten by a later refit. No leakage: only UNPLAYED fixtures are frozen (an
unplayed game cannot be in the training data), and each fixture is recorded once.

    python src/track.py freeze              # lock predictions for upcoming WC fixtures (next 3 days)
    python src/track.py freeze --date 2026-06-22
    python src/track.py grade               # fill actual results + correctness for played games
    python src/track.py report              # accuracy scoreboard (outcome / score / O-U / BTTS + Brier/log loss)

Record lives at data/predictions.csv (tracked in git — it's the accuracy record).
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from dixon_coles import load_results
from predict import load_model
import markets

PRED_PATH = Path(__file__).resolve().parent.parent / "data" / "predictions.csv"
COLS = ["predicted_at", "model_fitted_on", "date", "home_team", "away_team", "neutral",
        "p_home", "p_draw", "p_away", "pred_outcome", "pred_score", "exp_home", "exp_away",
        "p_over25", "p_btts_yes", "actual_home", "actual_away", "actual_outcome",
        "outcome_correct", "score_correct", "ou_correct", "btts_correct", "result"]


def _read() -> pd.DataFrame:
    if PRED_PATH.exists():
        df = pd.read_csv(PRED_PATH)
        for c in COLS:
            if c not in df.columns:
                df[c] = ""
        return df[COLS]
    return pd.DataFrame(columns=COLS)


def _write(df: pd.DataFrame) -> None:
    df.to_csv(PRED_PATH, index=False)


def _exists(df: pd.DataFrame, d: str, h: str, a: str) -> bool:
    if df.empty:
        return False
    return bool(((df["date"] == d) & (df["home_team"] == h) & (df["away_team"] == a)).any())


def freeze_cmd(args) -> None:
    m = load_model()
    results = load_results()
    results["date"] = pd.to_datetime(results["date"])
    wc = results[results["tournament"] == "FIFA World Cup"]
    unplayed = wc[wc["home_score"].isna()].copy()      # unplayed => not in training => no leakage
    if args.date:
        unplayed = unplayed[unplayed["date"] == pd.Timestamp(args.date)]
    else:
        start = pd.Timestamp(date.today())
        unplayed = unplayed[(unplayed["date"] >= start)
                            & (unplayed["date"] < start + pd.Timedelta(days=args.days))]

    df = _read()
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    new_rows = []
    skipped = unpriced = 0
    for _, r in unplayed.sort_values("date").iterrows():
        h, a = r["home_team"], r["away_team"]
        d = r["date"].strftime("%Y-%m-%d")
        if _exists(df, d, h, a):
            skipped += 1
            continue
        neutral = str(r["neutral"]).upper() == "TRUE"
        try:
            mat = m.score_matrix(h, a, neutral=neutral)
            lh, la = m.lambdas(h, a, neutral)
        except KeyError:
            unpriced += 1
            continue
        o = markets.outcome_probs(mat)
        over = m.over_prob(h, a, neutral=neutral, line=2.5, calibrated=True)
        bt = markets.btts(mat)
        score, _ = markets.correct_score(mat, 1)[0]
        row = {c: "" for c in COLS}
        row.update({
            "predicted_at": now, "model_fitted_on": m.fitted_on, "date": d,
            "home_team": h, "away_team": a, "neutral": neutral,
            "p_home": round(o["home"], 4), "p_draw": round(o["draw"], 4),
            "p_away": round(o["away"], 4),
            "pred_outcome": max(("home", "draw", "away"), key=lambda k: o[k]),
            "pred_score": score, "exp_home": round(lh, 2), "exp_away": round(la, 2),
            "p_over25": round(float(over), 4), "p_btts_yes": round(bt["btts_yes"], 4),
            "result": "pending",
        })
        new_rows.append(row)

    if new_rows:
        add = pd.DataFrame(new_rows, columns=COLS)
        df = add if df.empty else pd.concat([df, add], ignore_index=True)
        _write(df)
    extra = f"; {unpriced} unpriced (team below min-matches)" if unpriced else ""
    print(f"froze {len(new_rows)} prediction(s); {skipped} already tracked{extra}. "
          f"total tracked: {len(df)} ({int((df['result'] == 'pending').sum())} pending).")


def _live_scores(teams: list[str]) -> dict:
    """Completed scores from The Odds API scores endpoint (best-effort, ~2 credits),
    keyed by (home, away) in dataset names. Lets the tracker grade games before the
    martj42 dataset (which lags a day or two) has ingested them. {} if no key / fails."""
    try:
        import os
        import odds  # lazy: odds imports predict/markets; avoid an import cycle
        odds.load_dotenv()
        if not os.environ.get("ODDS_API_KEY"):
            return {}
        sport = odds.discover_wc_key()
        if not sport:
            return {}
        data, _ = odds.api_get(f"/sports/{sport}/scores/", daysFrom=3, dateFormat="iso")
        cache: dict = {}
        out: dict = {}
        for e in data:
            if not e.get("completed"):
                continue
            sc = {s["name"]: s["score"] for s in (e.get("scores") or [])}
            if e["home_team"] not in sc or e["away_team"] not in sc:
                continue
            h = odds.normalize(e["home_team"], teams, cache)
            a = odds.normalize(e["away_team"], teams, cache)
            if h is None or a is None:
                continue
            out[(h, a)] = (int(sc[e["home_team"]]), int(sc[e["away_team"]]))
        return out
    except (Exception, SystemExit):
        return {}


def grade_cmd(args) -> None:
    df = _read()
    if df.empty:
        print("No predictions tracked yet. Run `track.py freeze` first.")
        return
    results = load_results()
    results["date"] = pd.to_datetime(results["date"])
    played = results.dropna(subset=["home_score", "away_score"])
    # key on (home, away) — within one tournament a pairing is unique, and this is
    # robust to date discrepancies between the fixture list and the score source.
    pair_score = {(h, a): (int(hs), int(as_)) for h, a, hs, as_ in
                  zip(played["home_team"], played["away_team"],
                      played["home_score"], played["away_score"])}
    src = "dataset"
    if getattr(args, "live", False):
        live = _live_scores(load_model().teams)
        pair_score.update(live)   # live Odds API scores fill games the dataset still lags
        if live:
            src = f"dataset + {len(live)} live"

    # ensure the columns we fill are object dtype so writing ints/strings into cells
    # CSV-loaded as float/NaN doesn't trip pandas' incompatible-dtype warning
    fill = ["actual_home", "actual_away", "actual_outcome", "outcome_correct",
            "score_correct", "ou_correct", "btts_correct", "result"]
    df[fill] = df[fill].astype(object)

    # invariant: a prediction dated in the future cannot be settled. Guards against a
    # mismatched/stale score grading a game that hasn't happened yet; reset any such row.
    today = pd.Timestamp(date.today())
    future = (pd.to_datetime(df["date"]) > today) & (df["result"] == "graded")
    if future.any():
        df.loc[future, fill[:-1]] = ""
        df.loc[future, "result"] = "pending"
        print(f"reset {int(future.sum())} future-dated row(s) graded off mismatched scores.")

    graded = 0
    for i, r in df.iterrows():
        if str(r["result"]) != "pending":
            continue
        if pd.Timestamp(r["date"]) > today:        # don't settle games that haven't happened
            continue
        key = (r["home_team"], r["away_team"])
        if key not in pair_score:
            continue
        hs, as_ = pair_score[key]
        total = hs + as_
        ao = "home" if hs > as_ else ("draw" if hs == as_ else "away")
        df.at[i, "actual_home"] = hs
        df.at[i, "actual_away"] = as_
        df.at[i, "actual_outcome"] = ao
        df.at[i, "outcome_correct"] = int(r["pred_outcome"] == ao)
        df.at[i, "score_correct"] = int(str(r["pred_score"]) == f"{hs}-{as_}")
        df.at[i, "ou_correct"] = int((float(r["p_over25"]) > 0.5) == (total > 2.5))
        df.at[i, "btts_correct"] = int((float(r["p_btts_yes"]) > 0.5) == (hs > 0 and as_ > 0))
        df.at[i, "result"] = "graded"
        graded += 1
    _write(df)
    print(f"graded {graded} ({src}); {int((df['result'] == 'pending').sum())} still "
          f"pending (game not played / not in data).")


def report_cmd(args) -> None:
    df = _read()
    g = df[df["result"] == "graded"].copy()
    print(f"=== Prediction tracker ===  ({len(df)} tracked, {len(g)} graded)\n")
    if g.empty:
        print("Nothing graded yet -- run `track.py grade` after games are played.")
        return

    n = len(g)
    oc = g["outcome_correct"].astype(int).sum()
    sc = g["score_correct"].astype(int).sum()
    ouc = g["ou_correct"].astype(int).sum()
    bc = g["btts_correct"].astype(int).sum()
    print(f"  1X2 outcome      {oc}/{n}  ({oc / n * 100:.0f}%)")
    print(f"  exact score      {sc}/{n}  ({sc / n * 100:.0f}%)")
    print(f"  O/U 2.5          {ouc}/{n}  ({ouc / n * 100:.0f}%)")
    print(f"  BTTS             {bc}/{n}  ({bc / n * 100:.0f}%)")

    P = g[["p_home", "p_draw", "p_away"]].astype(float).to_numpy()
    idx = g["actual_outcome"].map({"home": 0, "draw": 1, "away": 2}).to_numpy()
    onehot = np.zeros_like(P)
    onehot[np.arange(n), idx] = 1.0
    brier = float(((P - onehot) ** 2).sum(axis=1).mean())
    logloss = float(-np.log(np.clip(P[np.arange(n), idx], 1e-12, 1.0)).mean())
    pick_p = float(P.max(axis=1).mean())
    print(f"  1X2 Brier        {brier:.3f}   log loss {logloss:.3f}")
    print(f"  pick calibration model avg {pick_p * 100:.0f}%  vs realised "
          f"{oc / n * 100:.0f}% correct")

    print("\n  graded results (pred vs actual):")
    for _, r in g.sort_values("date").iterrows():
        mark = "OK " if int(r["outcome_correct"]) else "X  "
        pick_prob = max(float(r["p_home"]), float(r["p_draw"]), float(r["p_away"]))
        print(f"    {mark} {r['date']}  {r['home_team']} {int(r['actual_home'])}-"
              f"{int(r['actual_away'])} {r['away_team']}   "
              f"(pred {r['pred_score']}, {r['pred_outcome']} {pick_prob * 100:.0f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description="Forward prediction tracker (freeze / grade / report)")
    sub = p.add_subparsers(dest="cmd", required=True)

    fz = sub.add_parser("freeze", help="lock point-in-time predictions for upcoming fixtures")
    fz.add_argument("--days", type=int, default=3, help="window from today (default 3)")
    fz.add_argument("--date", help="freeze a single date YYYY-MM-DD instead of the window")
    fz.set_defaults(func=freeze_cmd)

    gr = sub.add_parser("grade", help="fill actuals + correctness from played games")
    gr.add_argument("--live", action="store_true",
                    help="also pull completed scores from The Odds API (~2 credits) "
                         "when the dataset still lags")
    gr.set_defaults(func=grade_cmd)
    sub.add_parser("report", help="accuracy scoreboard").set_defaults(func=report_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
