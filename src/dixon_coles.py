"""Dixon-Coles bivariate-Poisson goals model for international football.

Each team gets an attack strength and a defence strength; a global home-advantage
term applies only at non-neutral venues; a low-score dependence parameter (rho)
corrects the independent-Poisson assumption for 0-0/1-0/0-1/1-1 scores.

Fitting is weighted MLE:
  - exponential time decay so recent matches dominate (Dixon-Coles 1997),
  - competition-importance weights so friendlies count less than tournament games.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Relative weight of a match by competition. Tournament knockouts/finals are the
# truest signal; friendlies are noisy (experimental line-ups, low stakes).
COMPETITION_WEIGHTS = {
    "FIFA World Cup": 1.0,
    "Copa América": 1.0,
    "UEFA Euro": 1.0,
    "African Cup of Nations": 0.95,
    "AFC Asian Cup": 0.95,
    "Gold Cup": 0.9,
    "FIFA World Cup qualification": 0.9,
    "UEFA Euro qualification": 0.9,
    "UEFA Nations League": 0.85,
    "CONCACAF Nations League": 0.8,
    "Confederations Cup": 0.85,
    "Friendly": 0.5,
}
DEFAULT_COMPETITION_WEIGHT = 0.7  # other qualifiers / minor tournaments


def competition_weight(tournament: str) -> float:
    if tournament in COMPETITION_WEIGHTS:
        return COMPETITION_WEIGHTS[tournament]
    for key, w in COMPETITION_WEIGHTS.items():
        if key.endswith("qualification") and key.split()[0] in tournament:
            return w
    return DEFAULT_COMPETITION_WEIGHT


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _tau(h, a, lam_h, lam_a, rho):
    """Dixon-Coles low-score correction, vectorised over arrays of (h, a)."""
    t = np.ones_like(lam_h, dtype=float)
    t = np.where((h == 0) & (a == 0), 1.0 - lam_h * lam_a * rho, t)
    t = np.where((h == 0) & (a == 1), 1.0 + lam_h * rho, t)
    t = np.where((h == 1) & (a == 0), 1.0 + lam_a * rho, t)
    t = np.where((h == 1) & (a == 1), 1.0 - rho, t)
    return t


@dataclass
class DixonColes:
    half_life_days: float = 730.0       # time-decay half-life (~2 years)
    min_matches: int = 8                # drop teams with too little recent data
    train_since: str = "2006-01-01"     # ignore ancient history for speed/relevance

    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.0
    rho: float = 0.0
    # Platt recalibration for the over/under market: p_cal = sigmoid(a + b*logit(p_raw)).
    # None means uncalibrated (raw Poisson). Fit out of sample by calibrate.py.
    totals_calib: tuple[float, float] | None = None
    teams: list[str] = field(default_factory=list)
    fitted_on: str = ""
    n_matches: int = 0

    # ---- data prep -------------------------------------------------------
    def _prepare(self, df: pd.DataFrame, as_of: date) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= pd.Timestamp(self.train_since)]
        df = df[df["date"] <= pd.Timestamp(as_of)]
        df = df.dropna(subset=["home_score", "away_score"])
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        # keep only teams with enough matches in the window
        counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        keep = set(counts[counts >= self.min_matches].index)
        df = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)]

        age_days = (pd.Timestamp(as_of) - df["date"]).dt.days.clip(lower=0)
        xi = np.log(2) / self.half_life_days
        time_w = np.exp(-xi * age_days)
        comp_w = df["tournament"].map(competition_weight).fillna(DEFAULT_COMPETITION_WEIGHT)
        df["weight"] = time_w * comp_w
        df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
        return df.reset_index(drop=True)

    # ---- fit -------------------------------------------------------------
    def fit(self, df: pd.DataFrame, as_of: date | None = None) -> "DixonColes":
        as_of = as_of or date.today()
        data = self._prepare(df, as_of)
        if data.empty:
            raise ValueError("No matches left after filtering; loosen train_since/min_matches.")

        teams = sorted(set(data["home_team"]) | set(data["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = data["home_team"].map(idx).to_numpy()
        ai = data["away_team"].map(idx).to_numpy()
        hs = data["home_score"].to_numpy()
        as_ = data["away_score"].to_numpy()
        w = data["weight"].to_numpy()
        non_neutral = (~data["neutral"]).to_numpy().astype(float)

        # params: [attack(n), defence(n), home_adv, rho]
        def unpack(p):
            return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]

        def neg_log_lik(p):
            atk, dfc, home, rho = unpack(p)
            log_lam_h = atk[hi] + dfc[ai] + home * non_neutral
            log_lam_a = atk[ai] + dfc[hi]
            lam_h = np.exp(log_lam_h)
            lam_a = np.exp(log_lam_a)
            tau = _tau(hs, as_, lam_h, lam_a, rho)
            tau = np.clip(tau, 1e-9, None)  # guard against negative tau
            ll = np.log(tau) + (hs * log_lam_h - lam_h) + (as_ * log_lam_a - lam_a)
            penalty = 100.0 * atk.sum() ** 2  # identifiability: mean attack ~ 0
            return -(w * ll).sum() + penalty

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [0.0]])
        bounds = [(-3, 3)] * n + [(-3, 3)] * n + [(-1, 1), (-0.2, 0.2)]
        res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "maxfun": 100000})

        atk, dfc, home, rho = unpack(res.x)
        self.attack = {t: float(atk[i]) for t, i in idx.items()}
        self.defence = {t: float(dfc[i]) for t, i in idx.items()}
        self.home_adv = float(home)
        self.rho = float(rho)
        self.teams = teams
        self.fitted_on = str(as_of)
        self.n_matches = len(data)
        # totals_calib (over/under recalibration) is fit out of sample by calibrate.py,
        # not here: the miscalibration only shows up on unseen matches.
        return self

    # ---- prediction ------------------------------------------------------
    def lambdas(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        for t in (home, away):
            if t not in self.attack:
                raise KeyError(f"Unknown / insufficiently-sampled team: {t!r}. "
                               f"Try `list_teams` to see what's available.")
        h = 1.0 if not neutral else 0.0
        lam_h = np.exp(self.attack[home] + self.defence[away] + self.home_adv * h)
        lam_a = np.exp(self.attack[away] + self.defence[home])
        return float(lam_h), float(lam_a)

    def over_prob(self, home: str, away: str, neutral: bool = True,
                  line: float = 2.5, calibrated: bool = True) -> float:
        """P(total goals > line). Raw value is the Poisson-sum tail; with `calibrated`
        and a fitted Platt map, it is corrected for the model's totals overconfidence."""
        lam_h, lam_a = self.lambdas(home, away, neutral)
        mu = lam_h + lam_a
        k = int(np.floor(line))  # e.g. line 2.5 -> need total >= 3
        from scipy.stats import poisson
        raw = float(1.0 - poisson.cdf(k, mu))
        if calibrated and self.totals_calib is not None:
            a, b = self.totals_calib
            return float(_sigmoid(a + b * _logit(raw)))
        return raw

    def score_matrix(self, home: str, away: str, neutral: bool = True,
                     max_goals: int = 10) -> np.ndarray:
        """P(home=i, away=j) for i,j in [0, max_goals], with DC correction."""
        lam_h, lam_a = self.lambdas(home, away, neutral)
        gh = np.arange(max_goals + 1)
        # independent Poisson outer product
        from scipy.stats import poisson
        ph = poisson.pmf(gh, lam_h)
        pa = poisson.pmf(gh, lam_a)
        mat = np.outer(ph, pa)
        # apply DC correction to the four low-score cells
        for h in (0, 1):
            for a in (0, 1):
                mat[h, a] *= float(_tau(np.array(h), np.array(a),
                                        np.array(lam_h), np.array(lam_a), self.rho))
        mat /= mat.sum()  # renormalise after correction
        return mat

    def rankings(self, top: int = 20) -> pd.DataFrame:
        """Teams ranked by overall strength (attack - defence; lower defence = better)."""
        rows = [{"team": t, "attack": self.attack[t], "defence": self.defence[t],
                 "rating": self.attack[t] - self.defence[t]} for t in self.teams]
        return (pd.DataFrame(rows).sort_values("rating", ascending=False)
                .head(top).reset_index(drop=True))


def load_results() -> pd.DataFrame:
    path = DATA_DIR / "results.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python src/download_data.py` first.")
    return pd.read_csv(path)
