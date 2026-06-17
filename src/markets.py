"""Turn a Dixon-Coles score matrix into betting-market probabilities.

Everything here is a deterministic function of the score matrix P(home=i, away=j),
so one fitted model generates every market for free.
"""
from __future__ import annotations

import numpy as np


def outcome_probs(mat: np.ndarray) -> dict[str, float]:
    """1X2: home win / draw / away win."""
    home = np.tril(mat, -1).sum()   # i > j
    draw = np.trace(mat)            # i == j
    away = np.triu(mat, 1).sum()    # i < j
    return {"home": float(home), "draw": float(draw), "away": float(away)}


def over_under(mat: np.ndarray, line: float = 2.5) -> dict[str, float]:
    n = mat.shape[0]
    totals = np.add.outer(np.arange(n), np.arange(n))
    over = float(mat[totals > line].sum())
    return {f"over_{line}": over, f"under_{line}": 1.0 - over}


def btts(mat: np.ndarray) -> dict[str, float]:
    """Both teams to score."""
    yes = float(mat[1:, 1:].sum())
    return {"btts_yes": yes, "btts_no": 1.0 - yes}


def asian_handicap(mat: np.ndarray, line: float = -0.5) -> dict[str, float]:
    """Home team handicap (e.g. -0.5 means home must win). Whole/half lines only;
    no quarter-line split or push handling beyond integer pushes returned separately."""
    n = mat.shape[0]
    margin = np.subtract.outer(np.arange(n), np.arange(n))  # home - away
    home_cover = float(mat[margin + line > 0].sum())
    push = float(mat[margin + line == 0].sum())
    away_cover = 1.0 - home_cover - push
    return {f"home_{line:+g}": home_cover, "push": push, f"away_{-line:+g}": away_cover}


def correct_score(mat: np.ndarray, top: int = 6) -> list[tuple[str, float]]:
    n = mat.shape[0]
    cells = [(f"{i}-{j}", float(mat[i, j])) for i in range(n) for j in range(n)]
    cells.sort(key=lambda kv: kv[1], reverse=True)
    return cells[:top]


def fair_odds(prob: float) -> float:
    """Decimal odds with no margin. Inf-guarded."""
    return float("inf") if prob <= 0 else round(1.0 / prob, 2)


def market_report(mat: np.ndarray) -> dict:
    """Everything, bundled — for a single match."""
    o = outcome_probs(mat)
    return {
        "outcome": o,
        "over_under_2.5": over_under(mat, 2.5),
        "over_under_3.5": over_under(mat, 3.5),
        "btts": btts(mat),
        "handicap_home_-0.5": asian_handicap(mat, -0.5),
        "handicap_home_-1.5": asian_handicap(mat, -1.5),
        "correct_score_top6": correct_score(mat, 6),
        "expected_goals": {
            "home": float((mat.sum(1) * np.arange(mat.shape[0])).sum()),
            "away": float((mat.sum(0) * np.arange(mat.shape[1])).sum()),
        },
    }


def value_vs_odds(model_prob: float, decimal_odds: float) -> dict:
    """Compare model probability to a posted decimal price → edge / EV.

    edge = model_prob - implied_prob; ev = model_prob * (odds - 1) - (1 - model_prob).
    Positive ev means +EV at this price.
    """
    implied = 1.0 / decimal_odds
    ev = model_prob * (decimal_odds - 1) - (1 - model_prob)
    return {
        "model_prob": round(model_prob, 4),
        "implied_prob": round(implied, 4),
        "edge": round(model_prob - implied, 4),
        "ev_per_unit": round(ev, 4),
        "value": ev > 0,
    }
