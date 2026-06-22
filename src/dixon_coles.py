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

import elo

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

    # Optional point-in-time Elo covariate (off by default; an opt-in experiment judged
    # by backtest.py). When on, the log goal-rates get a +/- gamma*(elo_diff/400) term,
    # so a single global coefficient pools strength info across all teams — most useful
    # for thin-data teams the per-team attack/defence can't pin down.
    use_elo: bool = False
    elo_gamma: float = 0.0
    elo_ratings: dict[str, float] = field(default_factory=dict)
    # Elo-as-PRIOR (the follow-up experiment): instead of a parallel covariate, add an
    # L2 penalty pulling each team's net strength (attack-defence) toward its Elo rating.
    # Because a thin-data team's likelihood barely constrains its params, the prior
    # dominates *for them* — adaptive shrinkage that targets the documented weakness
    # (strong/low-data teams shrunk to the mean) without disturbing well-sampled teams.
    elo_prior: bool = False
    elo_prior_lambda: float = 5.0
    # Squad-value prior (free Transfermarkt market value via squad.py): same shrinkage
    # mechanism, target = standardised log squad market value instead of Elo.
    squad_prior: bool = False
    squad_prior_lambda: float = 5.0
    # Pre-match CONTEXT covariates (rest/congestion/travel/climate via context.py): each
    # is a standardised home-minus-away advantage that shifts goal supremacy. Columns must
    # be attached to the df (context.attach) before fit. Predict-time values via lambdas(ctx=).
    use_context: bool = False
    context_cols: tuple[str, ...] = ("rest_adv", "cong_adv", "travel_adv", "climate_adv")
    context_theta: list[float] = field(default_factory=list)
    context_mean: list[float] = field(default_factory=list)
    context_std: list[float] = field(default_factory=list)

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
        if self.use_elo or self.elo_prior:     # point-in-time ratings as of `as_of`
            df, self.elo_ratings = elo.attach(df, as_of)
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
        if self.use_elo:
            ep = data[["home_elo_pre", "away_elo_pre"]].fillna(elo.START_RATING)
            elo_diff = (ep["home_elo_pre"] - ep["away_elo_pre"]).to_numpy() / 400.0
        else:
            elo_diff = np.zeros(len(data))
        # context covariates (standardised); columns attached by context.attach beforehand
        if self.use_context:
            Craw = data[list(self.context_cols)].to_numpy(dtype=float)
            self.context_mean = np.nanmean(Craw, axis=0).tolist()
            self.context_std = (np.nanstd(Craw, axis=0) + 1e-9).tolist()
            C = np.nan_to_num((Craw - self.context_mean) / np.array(self.context_std), nan=0.0)
            kctx = C.shape[1]
        else:
            C, kctx = None, 0
        # Unified strength PRIOR: shrink (attack - defence) toward a standardised
        # per-team target (Elo ratings, or squad market value). prior_z may contain
        # NaN for teams with no target (e.g. unmapped/unvalued nations) -> excluded.
        prior_z, prior_lambda = None, 0.0
        if self.elo_prior:
            ev = np.array([self.elo_ratings.get(t, elo.START_RATING) for t in teams])
            prior_z = (ev - ev.mean()) / (ev.std() + 1e-9)
            prior_lambda = self.elo_prior_lambda
        elif self.squad_prior:
            import warnings
            try:
                import squad
                sidx = squad.squad_index(teams, as_of)
            except Exception as e:                  # network/data failure -> degrade, don't crash
                warnings.warn(f"squad_prior disabled (squad data unavailable): {e}")
                sidx = {}
            sv = np.array([sidx.get(t, np.nan) for t in teams])
            m = ~np.isnan(sv)
            if m.sum() >= 2:
                z = np.full(len(teams), np.nan)
                z[m] = (sv[m] - sv[m].mean()) / (sv[m].std() + 1e-9)
                prior_z, prior_lambda = z, self.squad_prior_lambda
            else:
                warnings.warn("squad_prior requested but <2 teams have squad values; "
                              "prior is inactive (running as the plain model).")

        # params: [attack(n), defence(n), home_adv, rho, (extra if has_extra), (ctx_theta(kctx))]
        has_extra = self.use_elo or (prior_z is not None)
        prior_mask = ~np.isnan(prior_z) if prior_z is not None else None
        base = 2 * n + 2

        def unpack(p):
            extra = p[base] if has_extra else 0.0
            o = base + (1 if has_extra else 0)
            cth = p[o:o + kctx] if kctx else np.zeros(0)
            return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1], extra, cth

        def neg_log_lik(p):
            atk, dfc, home, rho, extra, cth = unpack(p)
            gamma = extra if self.use_elo else 0.0
            shift = (C @ cth) if kctx else 0.0
            log_lam_h = atk[hi] + dfc[ai] + home * non_neutral + gamma * elo_diff + shift
            log_lam_a = atk[ai] + dfc[hi] - gamma * elo_diff - shift
            lam_h = np.exp(log_lam_h)
            lam_a = np.exp(log_lam_a)
            tau = _tau(hs, as_, lam_h, lam_a, rho)
            tau = np.clip(tau, 1e-9, None)  # guard against negative tau
            ll = np.log(tau) + (hs * log_lam_h - lam_h) + (as_ * log_lam_a - lam_a)
            penalty = 100.0 * atk.sum() ** 2  # identifiability: mean attack ~ 0
            if prior_z is not None:           # shrink net strength toward beta*target
                d = atk[prior_mask] - dfc[prior_mask] - extra * prior_z[prior_mask]
                penalty += prior_lambda * np.sum(d ** 2)
            return -(w * ll).sum() + penalty

        parts = [np.zeros(n), np.zeros(n), np.array([0.25]), np.array([0.0])]
        bounds = [(-3, 3)] * n + [(-3, 3)] * n + [(-1, 1), (-0.2, 0.2)]
        if has_extra:
            parts.append(np.array([0.0]))
            bounds.append((-3, 3))            # gamma (covariate) or beta (prior)
        if kctx:
            parts.append(np.zeros(kctx))
            bounds += [(-1, 1)] * kctx        # context coefficients
        res = minimize(neg_log_lik, np.concatenate(parts), method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 500, "maxfun": 100000})

        atk, dfc, home, rho, extra, cth = unpack(res.x)
        self.attack = {t: float(atk[i]) for t, i in idx.items()}
        self.defence = {t: float(dfc[i]) for t, i in idx.items()}
        self.home_adv = float(home)
        self.rho = float(rho)
        if self.use_elo:
            self.elo_gamma = float(extra)     # prior's beta isn't needed at predict time
        if self.use_context:
            self.context_theta = cth.tolist()
        self.teams = teams
        self.fitted_on = str(as_of)
        self.n_matches = len(data)
        # totals_calib (over/under recalibration) is fit out of sample by calibrate.py,
        # not here: the miscalibration only shows up on unseen matches.
        return self

    # ---- prediction ------------------------------------------------------
    def lambdas(self, home: str, away: str, neutral: bool = True,
                ctx=None) -> tuple[float, float]:
        for t in (home, away):
            if t not in self.attack:
                raise KeyError(f"Unknown / insufficiently-sampled team: {t!r}. "
                               f"Try `list_teams` to see what's available.")
        h = 1.0 if not neutral else 0.0
        base_h = self.attack[home] + self.defence[away] + self.home_adv * h
        base_a = self.attack[away] + self.defence[home]
        if self.use_elo and self.elo_gamma:
            d = (self.elo_ratings.get(home, elo.START_RATING)
                 - self.elo_ratings.get(away, elo.START_RATING)) / 400.0
            base_h += self.elo_gamma * d
            base_a -= self.elo_gamma * d
        if self.use_context and ctx is not None and self.context_theta:
            cz = np.nan_to_num((np.asarray(ctx, float) - self.context_mean)
                               / np.asarray(self.context_std), nan=0.0)
            s = float(np.dot(cz, self.context_theta))
            base_h += s
            base_a -= s
        return float(np.exp(base_h)), float(np.exp(base_a))

    def over_prob(self, home: str, away: str, neutral: bool = True,
                  line: float = 2.5, calibrated: bool = True, ctx=None) -> float:
        """P(total goals > line). Raw value is the Poisson-sum tail; with `calibrated`
        and a fitted Platt map, it is corrected for the model's totals overconfidence."""
        lam_h, lam_a = self.lambdas(home, away, neutral, ctx=ctx)
        mu = lam_h + lam_a
        k = int(np.floor(line))  # e.g. line 2.5 -> need total >= 3
        from scipy.stats import poisson
        raw = float(1.0 - poisson.cdf(k, mu))
        if calibrated and self.totals_calib is not None:
            a, b = self.totals_calib
            return float(_sigmoid(a + b * _logit(raw)))
        return raw

    def score_matrix(self, home: str, away: str, neutral: bool = True,
                     max_goals: int = 10, ctx=None) -> np.ndarray:
        """P(home=i, away=j) for i,j in [0, max_goals], with DC correction."""
        lam_h, lam_a = self.lambdas(home, away, neutral, ctx=ctx)
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
