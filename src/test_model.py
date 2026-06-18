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


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
