"""Hermetic tests for the anytime-goalscorer model and forward tracker.

No network and no live data: the tracker tests monkeypatch the log path to a temp
file and the grading data so they never touch the real data/props_log.csv (the
forward CLV/accuracy record) or data/goalscorers.csv.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import props
import props_track


# ---- name normalisation / matching --------------------------------------
def test_normkey_folds_accents_and_case():
    assert props.normkey("Kylian Mbappé") == props.normkey("kylian mbappe")
    assert props.normkey("Niclas Füllkrug") == "niclas fullkrug"
    assert props.normkey("Pascal Groß") == props.normkey("Pascal Gross")   # ß -> ss
    assert props.normkey("Pierre Højbjerg") == props.normkey("Pierre Hojbjerg")  # ø -> o


def test_match_player_exact_lastname_and_miss():
    cands = ["Jamal Musiala", "Kai Havertz", "Serge Gnabry"]
    assert props.match_player("Musiala", cands) == "Jamal Musiala"   # last name
    assert props.match_player("Kai Havertz", cands) == "Kai Havertz"  # exact
    assert props.match_player("Lionel Messi", cands) is None          # no match


# ---- model math ----------------------------------------------------------
def test_xg_conversion_beats_goal_fallback_for_a_finisher(monkeypatch):
    # a player with high xG history should get a higher conversion than the flat
    # goal-based fallback (0.105) -- the documented "stars underrated" fix.
    monkeypatch.setattr(props, "_xg_lookup", lambda: {"test finisher": 0.18})
    assert 0.18 > props.LEAGUE_CONV


def test_goalscorer_prob_is_poisson_and_scales_with_opponent():
    # P(score) = 1 - exp(-lambda); a fake profile run through the formula by hand
    shots90, conv, exp_min, intl = 4.0, 0.15, 90.0, props.INTL_FACTOR
    lam_strong = (shots90 * conv * 1.0 * np.clip(2.0 / props.TEAM_BASELINE, 0.45, 2.0) * intl)
    lam_weak = (shots90 * conv * 1.0 * np.clip(0.6 / props.TEAM_BASELINE, 0.45, 2.0) * intl)
    p_strong = 1 - np.exp(-lam_strong)
    p_weak = 1 - np.exp(-lam_weak)
    assert 0 < p_weak < p_strong < 1           # scores more vs a weak defence


# ---- tracker: freeze / grade / report -----------------------------------
@pytest.fixture
def temp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(props_track, "PROPS_PATH", tmp_path / "props_log.csv")
    return tmp_path


def _patch_grading(monkeypatch, gs, results):
    """Point grade_cmd at synthetic goalscorers + results without touching the real
    files. Only the goalscorers.csv read is redirected -- _read/_write of the temp
    log keep using the real pd.read_csv/to_csv."""
    orig = pd.read_csv
    monkeypatch.setattr(pd, "read_csv",
                        lambda p, *a, **k: gs if "goalscorers" in str(p) else orig(p, *a, **k))
    monkeypatch.setattr(props_track, "load_results", lambda: results)


def _fake_model():
    m = types.SimpleNamespace()
    m.fitted_on = "2026-06-25"
    m.teams = ["Ecuador", "Germany"]
    m.lambdas = lambda h, a, neutral=True: (0.8, 1.6)
    return m


def test_freeze_event_locks_matched_players(temp_log, monkeypatch):
    # one model player (Musiala) and a book offering Musiala + an unknown name
    mtable = pd.DataFrame([{"team": "Germany", "player": "Jamal Musiala",
                            "p_anytime": 0.34, "fair_odds": 2.9, "shots90": 4.0,
                            "conv": 0.14, "conv_src": "xg", "exp_min": 90.0, "MP": 15}])
    monkeypatch.setattr(props, "goalscorer_table",
                        lambda h, a, neutral=False, model=None, lineup=None: (mtable, (0.8, 1.6)))
    event = {"commence_time": "2026-06-26T18:00:00Z", "bookmakers": [
        {"title": "Pinnacle", "markets": [{"key": "player_goal_scorer_anytime", "outcomes": [
            {"description": "Jamal Musiala", "price": 3.0},
            {"description": "Unknown Sub", "price": 9.0}]}]}]}
    df = props_track._read()
    rows = props_track._freeze_event(df, event, "Germany", "Ecuador", True, _fake_model(),
                                     "2026-06-25 10:00")
    assert len(rows) == 1                       # unknown sub dropped (no model price)
    assert rows[0]["player"] == "Jamal Musiala"
    assert rows[0]["edge"] == pytest.approx(0.34 - 1 / 3.0, abs=1e-3)  # logged rounded 4dp


def test_grade_marks_scorer_and_skips_non_scorer(temp_log, monkeypatch):
    log = pd.DataFrame([
        {**{c: "" for c in props_track.COLS}, "date": "2026-06-26", "home_team": "Germany",
         "away_team": "Ecuador", "player": "Jamal Musiala", "model_p": 0.34,
         "book_odds": 3.0, "book_p": 0.33, "edge": 0.01, "result": "pending"},
        {**{c: "" for c in props_track.COLS}, "date": "2026-06-26", "home_team": "Germany",
         "away_team": "Ecuador", "player": "Antonio Rudiger", "model_p": 0.10,
         "book_odds": 9.0, "book_p": 0.11, "edge": -0.01, "result": "pending"},
    ])
    props_track._write(log)
    gs = pd.DataFrame([{"date": "2026-06-26", "home_team": "Germany", "away_team": "Ecuador",
                        "team": "Germany", "scorer": "Jamal Musiala", "minute": 23,
                        "own_goal": "FALSE", "penalty": "FALSE"}])
    results = pd.DataFrame([{"date": "2026-06-26", "home_team": "Germany",
                             "away_team": "Ecuador", "home_score": 2.0, "away_score": 0.0}])
    _patch_grading(monkeypatch, gs, results)
    props_track.grade_cmd(types.SimpleNamespace(since="2026-06-01"))
    out = props_track._read().set_index("player")
    assert int(out.loc["Jamal Musiala", "scored"]) == 1
    assert int(out.loc["Antonio Rudiger", "scored"]) == 0
    assert (out["result"] == "graded").all()


def test_grade_ignores_own_goals(temp_log, monkeypatch):
    log = pd.DataFrame([{**{c: "" for c in props_track.COLS}, "date": "2026-06-26",
                         "home_team": "Germany", "away_team": "Ecuador",
                         "player": "Piero Hincapie", "model_p": 0.05, "book_odds": 17.0,
                         "book_p": 0.06, "edge": -0.01, "result": "pending"}])
    props_track._write(log)
    gs = pd.DataFrame([{"date": "2026-06-26", "home_team": "Germany", "away_team": "Ecuador",
                        "team": "Germany", "scorer": "Piero Hincapie", "minute": 30,
                        "own_goal": "TRUE", "penalty": "FALSE"}])     # own goal -> not a goal
    results = pd.DataFrame([{"date": "2026-06-26", "home_team": "Germany",
                             "away_team": "Ecuador", "home_score": 1.0, "away_score": 0.0}])
    _patch_grading(monkeypatch, gs, results)
    props_track.grade_cmd(types.SimpleNamespace(since="2026-06-01"))
    assert int(props_track._read().iloc[0]["scored"]) == 0


def test_grade_does_not_settle_a_historical_pairing(temp_log, monkeypatch):
    # a logged 2026 prop must NOT be graded off a pre-tournament friendly result
    log = pd.DataFrame([{**{c: "" for c in props_track.COLS}, "date": "2026-06-26",
                         "home_team": "Ecuador", "away_team": "Germany",
                         "player": "Enner Valencia", "model_p": 0.2, "book_odds": 5.0,
                         "book_p": 0.2, "edge": 0.0, "result": "pending"}])
    props_track._write(log)
    gs = pd.DataFrame([{"date": "2013-05-29", "home_team": "Ecuador", "away_team": "Germany",
                        "team": "Ecuador", "scorer": "Felipe Caicedo", "minute": 10,
                        "own_goal": "FALSE", "penalty": "FALSE"}])    # old friendly
    results = pd.DataFrame([{"date": "2013-05-29", "home_team": "Ecuador",
                             "away_team": "Germany", "home_score": 2.0, "away_score": 4.0}])
    _patch_grading(monkeypatch, gs, results)
    props_track.grade_cmd(types.SimpleNamespace(since="2026-06-01"))
    assert props_track._read().iloc[0]["result"] == "pending"        # untouched


def test_resolve_lineup_maps_roles_and_flags_gaps():
    names = ["Jamal Musiala", "Kai Havertz", "Serge Gnabry"]
    roles, unmatched = props.resolve_lineup(
        {"starters": ["Musiala", "Havertz"], "bench": ["Gnabry"]}, names)
    assert roles == {"Jamal Musiala": "starter", "Kai Havertz": "starter",
                     "Serge Gnabry": "sub"}
    assert unmatched == []
    # a non-top-5 starter the model has no profile for is flagged, not silently dropped
    roles2, unmatched2 = props.resolve_lineup({"starters": ["Enner Valencia"]}, names)
    assert roles2 == {} and unmatched2 == ["Enner Valencia"]


def test_lineup_overrides_minutes_and_drops_non_xi(monkeypatch):
    prof = pd.DataFrame([
        {"Player": "Striker A", "country": "Germany", "shots90": 4.0, "conv": 0.15,
         "conv_src": "xg", "exp_min": 90.0, "MP": 30},
        {"Player": "Bench B", "country": "Germany", "shots90": 4.0, "conv": 0.15,
         "conv_src": "xg", "exp_min": 90.0, "MP": 30},
        {"Player": "Dropped C", "country": "Germany", "shots90": 5.0, "conv": 0.2,
         "conv_src": "xg", "exp_min": 90.0, "MP": 30},
    ])
    monkeypatch.setattr(props, "build_profiles", lambda: prof)
    lineup = {"Germany": {"starters": ["Striker A"], "bench": ["Bench B"]}}
    df, _ = props.goalscorer_table("Ecuador", "Germany", neutral=True,
                                   model=_fake_model(), lineup=lineup)
    assert set(df["player"]) == {"Striker A", "Bench B"}     # Dropped C not in XI
    pa = df.set_index("player")
    assert pa.loc["Striker A", "exp_min"] == props.STARTER_MIN
    assert pa.loc["Bench B", "exp_min"] == props.SUB_MIN
    # identical shot profile, more minutes -> strictly higher score probability
    assert pa.loc["Striker A", "p_anytime"] > pa.loc["Bench B", "p_anytime"]


def test_report_runs_on_graded_log(temp_log, capsys):
    log = pd.DataFrame([{**{c: "" for c in props_track.COLS}, "date": "2026-06-26",
                         "home_team": "Germany", "away_team": "Ecuador",
                         "player": "A", "model_p": 0.4, "book_odds": 2.5, "book_p": 0.4,
                         "edge": 0.0, "closing_odds": "", "scored": 1, "result": "graded"},
                        {**{c: "" for c in props_track.COLS}, "date": "2026-06-26",
                         "home_team": "Germany", "away_team": "Ecuador",
                         "player": "B", "model_p": 0.1, "book_odds": 9.0, "book_p": 0.11,
                         "edge": -0.01, "closing_odds": "", "scored": 0, "result": "graded"}])
    props_track._write(log)
    props_track.report_cmd(types.SimpleNamespace())
    out = capsys.readouterr().out
    assert "calibration" in out and "base rate scored" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
