"""Hermetic sanity tests — no network, tiny synthetic fixture."""
import numpy as np
import pandas as pd

from dixon_coles import DixonColes, competition_weight
import markets


def _toy_df():
    # Team A clearly stronger than B; both play enough matches to survive filtering.
    rows = []
    for i in range(20):
        rows.append(("2024-01-01", "A", "B", 3, 0, "Friendly", False))
        rows.append(("2024-06-01", "B", "A", 0, 2, "Friendly", False))
        rows.append(("2025-01-01", "A", "C", 2, 1, "Friendly", False))
        rows.append(("2025-06-01", "C", "B", 1, 1, "Friendly", False))
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                       "home_score", "away_score", "tournament", "neutral"])


def test_fit_and_strength_ordering():
    m = DixonColes(train_since="2020-01-01", min_matches=5).fit(_toy_df())
    # A should outrate B
    assert (m.attack["A"] - m.defence["A"]) > (m.attack["B"] - m.defence["B"])


def test_score_matrix_is_a_distribution():
    m = DixonColes(train_since="2020-01-01", min_matches=5).fit(_toy_df())
    mat = m.score_matrix("A", "B", neutral=True)
    assert mat.shape == (11, 11)
    assert abs(mat.sum() - 1.0) < 1e-9
    assert (mat >= 0).all()


def test_markets_sum_to_one():
    m = DixonColes(train_since="2020-01-01", min_matches=5).fit(_toy_df())
    mat = m.score_matrix("A", "B", neutral=True)
    o = markets.outcome_probs(mat)
    assert abs(o["home"] + o["draw"] + o["away"] - 1.0) < 1e-9
    ou = markets.over_under(mat, 2.5)
    assert abs(ou["over_2.5"] + ou["under_2.5"] - 1.0) < 1e-9


def test_home_advantage_mechanism():
    # With a positive home_adv, a non-neutral venue must lift the home rate.
    m = DixonColes(train_since="2020-01-01", min_matches=5).fit(_toy_df())
    m.home_adv = 0.25  # force a known positive advantage
    lh_home, _ = m.lambdas("A", "B", neutral=False)
    lh_neutral, _ = m.lambdas("A", "B", neutral=True)
    assert lh_home > lh_neutral


def test_competition_weight_friendly_lt_worldcup():
    assert competition_weight("Friendly") < competition_weight("FIFA World Cup")


def test_platt_shrinks_toward_center():
    # b < 1 compresses predictions toward the base rate, curing overconfidence.
    m = DixonColes(train_since="2020-01-01", min_matches=5).fit(_toy_df())
    m.totals_calib = (0.0, 0.5)  # b=0.5 with a=0 shrinks logits toward 0 (p->0.5)
    for h, a in [("A", "B"), ("B", "A")]:
        raw = m.over_prob(h, a, neutral=True, calibrated=False)
        cal = m.over_prob(h, a, neutral=True, calibrated=True)
        assert abs(cal - 0.5) <= abs(raw - 0.5) + 1e-9


def test_platt_identity_equals_raw():
    m = DixonColes(train_since="2020-01-01", min_matches=5).fit(_toy_df())
    m.totals_calib = (0.0, 1.0)  # identity map
    raw = m.over_prob("A", "B", neutral=True, calibrated=False)
    cal = m.over_prob("A", "B", neutral=True, calibrated=True)
    assert abs(raw - cal) < 1e-9


def test_none_calib_equals_raw():
    m = DixonColes(train_since="2020-01-01", min_matches=5).fit(_toy_df())
    assert m.totals_calib is None
    raw = m.over_prob("A", "B", neutral=True, calibrated=False)
    cal = m.over_prob("A", "B", neutral=True, calibrated=True)
    assert abs(raw - cal) < 1e-9


def test_value_detection():
    v = markets.value_vs_odds(0.6, 2.0)  # model 60% vs implied 50%
    assert v["value"] and v["edge"] > 0
    v2 = markets.value_vs_odds(0.4, 2.0)
    assert not v2["value"]


def test_bet_grading():
    from slate import _won
    assert _won("home", "", 2, 0)[0] == "win"
    assert _won("home", "", 0, 1)[0] == "loss"
    assert _won("away", "", 0, 1)[0] == "win"
    assert _won("draw", "", 1, 1)[0] == "win"
    assert _won("draw", "", 1, 0)[0] == "loss"
    assert _won("over", 2.5, 2, 1)[0] == "win"      # 3 goals > 2.5
    assert _won("over", 2.5, 1, 1)[0] == "loss"     # 2 goals < 2.5
    assert _won("under", 2.5, 1, 1)[0] == "win"
    assert _won("under", 3.0, 1, 2)[0] == "push"    # exactly on an integer line


def test_line_key_matches_logged_lines():
    from odds import _line_key
    assert _line_key("home", "") == ""          # non-line market -> empty key
    assert _line_key("over", "2.5") == "2.5"     # logged string line
    assert _line_key("over", 2.5) == "2.5"       # scan-row float line
    assert _line_key("ah_away", "-1.0") == "-1.0"
    assert _line_key("ah_home", -1.0) == "-1.0"


def test_challenge_parlay_grades_and_pays():
    import tempfile, json
    from pathlib import Path
    from types import SimpleNamespace
    import challenge
    orig_log, orig_scores = challenge.LOG, challenge._scores
    try:
        challenge.LOG = Path(tempfile.mkdtemp()) / "challenge_log.csv"
        legs = [{"home": "France", "away": "Iraq", "market": "ah_away", "line": 2.5, "odds": 2.0},
                {"home": "Argentina", "away": "Austria", "market": "under", "line": 2.5, "odds": 1.5}]
        challenge.log_cmd(SimpleNamespace(date="2026-06-22", stake=25.0, legs=json.dumps(legs)))
        # France 2-0: Iraq +2.5 -> (0-2)+2.5 = +0.5 win. Argentina 1-0: total 1 < 2.5 under win.
        challenge._scores = lambda use_live: {("France", "Iraq"): (2, 0),
                                              ("Argentina", "Austria"): (1, 0)}
        challenge.grade_cmd(SimpleNamespace(no_live=True))
        df = challenge._read()
        assert (df["ticket_result"] == "win").all()
        assert abs(float(df["payout"].iloc[0]) - 25 * 2.0 * 1.5) < 1e-6   # both legs win -> 3x
    finally:
        challenge.LOG, challenge._scores = orig_log, orig_scores


def test_elo_is_point_in_time_and_orders_strength():
    import elo
    rows = [
        ("2024-01-01", "Strong", "Weak", 3, 0, "FIFA World Cup", True),
        ("2024-02-01", "Strong", "Weak", 2, 0, "FIFA World Cup", True),
        ("2024-03-01", "Weak", "Strong", 0, 1, "FIFA World Cup", True),
        ("2030-01-01", "Strong", "Weak", None, None, "FIFA World Cup", True),  # future/unplayed
    ]
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                     "home_score", "away_score", "tournament", "neutral"])
    out, ratings = elo.attach(df, as_of="2024-06-01")
    # first match: both teams start at the default, so pre-match ratings are equal
    first = out.sort_values("date").iloc[0]
    assert first["home_elo_pre"] == first["away_elo_pre"] == elo.START_RATING
    # after winning, Strong outrates Weak
    assert ratings["Strong"] > ratings["Weak"]
    # zero-sum: ratings stay centred on the start value
    assert abs((ratings["Strong"] + ratings["Weak"]) - 2 * elo.START_RATING) < 1e-6
    # point-in-time: the unplayed 2030 fixture never gets a pre-match rating
    assert out[out["date"] == "2030-01-01"]["home_elo_pre"].isna().all()


def test_track_grades_against_actual():
    # grade_cmd should fill actual result + per-market correctness for a played game.
    # Hermetic: temp PRED_PATH + stubbed load_results (no disk / network).
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    import track
    orig_path, orig_load = track.PRED_PATH, track.load_results
    try:
        track.PRED_PATH = Path(tempfile.mkdtemp()) / "predictions.csv"
        row = {c: "" for c in track.COLS}
        row.update({"date": "2026-06-21", "home_team": "A", "away_team": "B",
                    "neutral": True, "p_home": 0.6, "p_draw": 0.25, "p_away": 0.15,
                    "pred_outcome": "home", "pred_score": "2-0",
                    "p_over25": 0.6, "p_btts_yes": 0.3, "result": "pending"})
        track._write(pd.DataFrame([row], columns=track.COLS))
        played = pd.DataFrame(
            [("2026-06-21", "A", "B", 2, 0, "FIFA World Cup", True)],
            columns=["date", "home_team", "away_team", "home_score", "away_score",
                     "tournament", "neutral"])
        track.load_results = lambda: played
        track.grade_cmd(SimpleNamespace())
        r = track._read().iloc[0]
        assert r["result"] == "graded"
        assert int(r["outcome_correct"]) == 1   # home win predicted, home won
        assert int(r["score_correct"]) == 1     # predicted 2-0, actual 2-0
        assert int(r["ou_correct"]) == 0        # predicted over (0.6) but 2 goals -> under
        assert int(r["btts_correct"]) == 1      # predicted no (0.3 yes) and B didn't score
    finally:
        track.PRED_PATH, track.load_results = orig_path, orig_load


def test_ah_confidence_measures_distance_from_coinflip():
    from predict import _ah_confidence
    # confidence is about the favourite COVERING the handicap, i.e. how far the cover
    # probability sits from 50% — not about who wins the match.
    assert _ah_confidence(0.65) == "HIGH"    # cover 65% -> 0.15 from coin-flip
    assert _ah_confidence(0.35) == "HIGH"    # symmetric: 0.15 below 50% is just as decisive
    assert _ah_confidence(0.57) == "MEDIUM"  # 0.07 from 50%
    assert _ah_confidence(0.52) == "LOW"     # basically a coin-flip on the line


def test_lean_flag_marks_known_bias():
    from odds import lean_flag
    # h2h consensus: home is the favourite (0.6), away the underdog (0.15).
    nv = {("h2h", "Home"): 0.6, ("h2h", "Draw"): 0.25, ("h2h", "Away"): 0.15}
    assert lean_flag("draw", "", nv, "Home", "Away") == "draw"
    assert lean_flag("under", 2.5, nv, "Home", "Away") == "under"
    assert lean_flag("away", "", nv, "Home", "Away") == "dog"      # backing the underdog
    assert lean_flag("home", "", nv, "Home", "Away") == ""         # backing the favourite
    assert lean_flag("over", 2.5, nv, "Home", "Away") == ""        # not a low-score lean
    # Asian handicap: receiving points (+) = backing the relative underdog.
    assert lean_flag("ah_away", 1.5, nv, "Home", "Away") == "dog"
    assert lean_flag("ah_home", -1.5, nv, "Home", "Away") == ""    # laying points = favourite


def test_close_snapshots_best_price():
    # close_cmd should write the current best price into closing_odds for pending
    # bets, matching line markets by line and non-line markets by "". Hermetic:
    # temp LOG_PATH (never touch the real bet_log) + stubbed _gather (no network).
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    import slate, odds
    orig_path, orig_gather = slate.LOG_PATH, odds._gather
    try:
        slate.LOG_PATH = Path(tempfile.mkdtemp()) / "bet_log.csv"
        slate.record_bet("A", "B", "over", 2.10, 0.55, "2026-06-21", line=2.5, tier="HIGH")
        slate.record_bet("A", "B", "draw", 3.40, 0.30, "2026-06-21", tier="HIGH")
        matched = [[
            {"home": "A", "away": "B", "market": "over", "line": 2.5, "price": 1.95},
            {"home": "A", "away": "B", "market": "draw", "line": "", "price": 3.60},
        ]]
        odds._gather = lambda args: (None, "wc", [], matched, [],
                                     {"remaining": "1", "used": "1"})
        odds.close_cmd(SimpleNamespace(date="2026-06-21", sport=None, regions="us"))
        df = slate._read_log()
        over = df[df["market"] == "over"].iloc[0]
        draw = df[df["market"] == "draw"].iloc[0]
        assert abs(float(over["closing_odds"]) - 1.95) < 1e-9
        assert abs(float(draw["closing_odds"]) - 3.60) < 1e-9
    finally:
        slate.LOG_PATH, odds._gather = orig_path, orig_gather


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
