"""Walk-forward calibration backtest for the Dixon-Coles model.

The point of this is honesty: before trusting any model edge we need to know the
probabilities are *calibrated* out of sample. So we refit the model only on data
available at the time, predict held-out matches, and score the predictions with
proper scoring rules plus reliability curves.

No leakage: a match on date d is always scored by a model fit strictly before d.

Baseline = "climatology": the historical home/draw/away (and over/under) base
rates, split by neutral vs. non-neutral venue, computed from pre-test data only.
A model that can't beat climatology on log loss isn't adding information. (A true
*market* baseline needs historical odds we don't pay for yet — swap it in here
when an odds feed exists; the scoring harness is identical.)

Usage:
    python src/backtest.py --test-start 2022-06-01 --refit-days 180
    python src/backtest.py --competitive-only --no-plot
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from dixon_coles import DixonColes, load_results
import markets

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
FRIENDLY = "Friendly"


# ---- scoring primitives --------------------------------------------------
def log_loss(p_realized: float) -> float:
    return -np.log(np.clip(p_realized, 1e-12, 1.0))


def brier_multiclass(probs: np.ndarray, onehot: np.ndarray) -> float:
    return float(((probs - onehot) ** 2).sum())


def rps(probs_ordered: np.ndarray, outcome_idx: int) -> float:
    """Ranked probability score for ordered 1X2 (home>draw>away). Lower is better."""
    cum_pred = np.cumsum(probs_ordered)
    obs = np.zeros_like(probs_ordered)
    obs[outcome_idx] = 1.0
    cum_obs = np.cumsum(obs)
    r = len(probs_ordered)
    return float(((cum_pred[:-1] - cum_obs[:-1]) ** 2).sum() / (r - 1))


# ---- reliability ---------------------------------------------------------
class Reliability:
    """Pools (predicted_prob, occurred) pairs into bins for a calibration curve."""

    def __init__(self, n_bins: int = 10):
        self.edges = np.linspace(0, 1, n_bins + 1)
        self.sum_pred = np.zeros(n_bins)
        self.sum_obs = np.zeros(n_bins)
        self.count = np.zeros(n_bins)

    def add(self, prob: float, occurred: bool) -> None:
        b = min(np.searchsorted(self.edges, prob, side="right") - 1, len(self.count) - 1)
        b = max(b, 0)
        self.sum_pred[b] += prob
        self.sum_obs[b] += 1.0 if occurred else 0.0
        self.count[b] += 1

    def add_many(self, probs, occurred) -> None:
        for p, o in zip(probs, occurred):
            self.add(float(p), bool(o))

    def table(self) -> pd.DataFrame:
        mask = self.count > 0
        mean_pred = np.divide(self.sum_pred, self.count, where=mask, out=np.full_like(self.sum_pred, np.nan))
        emp = np.divide(self.sum_obs, self.count, where=mask, out=np.full_like(self.sum_obs, np.nan))
        return pd.DataFrame({
            "bin": [f"{self.edges[i]:.1f}-{self.edges[i+1]:.1f}" for i in range(len(self.count))],
            "n": self.count.astype(int),
            "mean_pred": mean_pred,
            "empirical": emp,
        })[mask]

    def ece(self) -> float:
        """Expected calibration error: count-weighted |pred - empirical|."""
        mask = self.count > 0
        if not mask.any():
            return float("nan")
        mean_pred = self.sum_pred[mask] / self.count[mask]
        emp = self.sum_obs[mask] / self.count[mask]
        w = self.count[mask] / self.count[mask].sum()
        return float((w * np.abs(mean_pred - emp)).sum())


# ---- baseline ------------------------------------------------------------
def climatology(df_train: pd.DataFrame) -> dict:
    """Base rates split by venue, from pre-test competitive data."""
    def rates(sub):
        h = (sub["home_score"] > sub["away_score"]).mean()
        d = (sub["home_score"] == sub["away_score"]).mean()
        a = (sub["home_score"] < sub["away_score"]).mean()
        over = ((sub["home_score"] + sub["away_score"]) > 2.5).mean()
        return {"home": h, "draw": d, "away": a, "over": over}
    neutral = df_train["neutral"]
    return {"neutral": rates(df_train[neutral]), "venue": rates(df_train[~neutral])}


# ---- backtest driver -----------------------------------------------------
def run(test_start: str, test_end: str, refit_days: int, competitive_only: bool,
        half_life: float, min_matches: int,
        totals_calib: tuple[float, float] | None = None, use_elo: bool = False,
        elo_prior: bool = False, elo_prior_lambda: float = 5.0) -> dict:
    """Walk-forward backtest. If `totals_calib` (Platt a, b) is given, the O/U
    market is also scored with the recalibration so raw vs. calibrated can be
    compared in a single pass."""
    df = load_results()
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    ts, te = pd.Timestamp(test_start), pd.Timestamp(test_end)
    if competitive_only:
        df = df[df["tournament"] != FRIENDLY]
    test = df[(df["date"] > ts) & (df["date"] <= te)].sort_values("date").reset_index(drop=True)
    base_rates = climatology(df[df["date"] <= ts])
    has_cal = totals_calib is not None

    rows = []
    totals = []  # OOS (mu, total) pairs for fitting the dispersion elsewhere
    rel_1x2 = Reliability(10)
    rel_ou = Reliability(10)        # raw Poisson totals
    rel_ou_cal = Reliability(10)    # NB-recalibrated totals

    model: DixonColes | None = None
    next_refit = ts
    fit_count = 0

    for _, r in test.iterrows():
        d = r["date"]
        # refit whenever we've crossed the next boundary; always fit strictly before d
        while d > next_refit:
            as_of = next_refit.date()
            model = DixonColes(half_life_days=half_life, min_matches=min_matches,
                               use_elo=use_elo, elo_prior=elo_prior,
                               elo_prior_lambda=elo_prior_lambda).fit(df, as_of=as_of)
            if has_cal:
                model.totals_calib = totals_calib
            fit_count += 1
            next_refit = next_refit + pd.Timedelta(days=refit_days)
        h, a = r["home_team"], r["away_team"]
        if model is None or h not in model.attack or a not in model.attack:
            rows.append({"scored": False})
            continue

        neutral = bool(r["neutral"])
        mat = model.score_matrix(h, a, neutral=neutral)
        o = markets.outcome_probs(mat)
        probs = np.array([o["home"], o["draw"], o["away"]])
        lam_h, lam_a = model.lambdas(h, a, neutral)
        mu = lam_h + lam_a
        over_raw = model.over_prob(h, a, neutral=neutral, line=2.5, calibrated=False)
        over_cal = model.over_prob(h, a, neutral=neutral, line=2.5, calibrated=True) if has_cal else None

        hs, as_ = r["home_score"], r["away_score"]
        outcome_idx = 0 if hs > as_ else (1 if hs == as_ else 2)
        total = int(hs + as_)
        over_occurred = total > 2.5

        br = base_rates["neutral"] if neutral else base_rates["venue"]
        base_probs = np.array([br["home"], br["draw"], br["away"]])
        base_probs = base_probs / base_probs.sum()

        onehot = np.zeros(3); onehot[outcome_idx] = 1.0
        rec = {
            "scored": True,
            "logloss": log_loss(probs[outcome_idx]),
            "rps": rps(probs, outcome_idx),
            "brier": brier_multiclass(probs, onehot),
            "correct": int(probs.argmax() == outcome_idx),
            "base_logloss": log_loss(base_probs[outcome_idx]),
            "base_rps": rps(base_probs, outcome_idx),
            "ou_logloss": log_loss(over_raw if over_occurred else 1 - over_raw),
            "ou_base_logloss": log_loss(br["over"] if over_occurred else 1 - br["over"]),
        }
        if has_cal:
            rec["ou_cal_logloss"] = log_loss(over_cal if over_occurred else 1 - over_cal)
            rel_ou_cal.add(over_cal, over_occurred)
        rows.append(rec)
        totals.append({"mu": mu, "total": total, "raw_over": over_raw,
                       "over_occurred": over_occurred})
        rel_1x2.add_many(probs, onehot.astype(bool))
        rel_ou.add(over_raw, over_occurred)

    res = pd.DataFrame(rows)
    scored = res[res["scored"]] if "scored" in res else res
    n_scored = int(scored.shape[0]) if not scored.empty else 0
    n_skipped = int((~res["scored"]).sum()) if "scored" in res else 0

    summary = {
        "test_window": f"{test_start} -> {test_end}",
        "competitive_only": competitive_only,
        "refits": fit_count,
        "n_scored": n_scored,
        "n_skipped_unknown_team": n_skipped,
    }
    if n_scored:
        summary.update({
            "model_logloss_1x2": float(scored["logloss"].mean()),
            "base_logloss_1x2": float(scored["base_logloss"].mean()),
            "model_rps": float(scored["rps"].mean()),
            "base_rps": float(scored["base_rps"].mean()),
            "model_brier_1x2": float(scored["brier"].mean()),
            "accuracy_1x2": float(scored["correct"].mean()),
            "model_logloss_ou25": float(scored["ou_logloss"].mean()),
            "base_logloss_ou25": float(scored["ou_base_logloss"].mean()),
            "ece_1x2": rel_1x2.ece(),
            "ece_ou25": rel_ou.ece(),
        })
        if has_cal:
            summary["model_logloss_ou25_cal"] = float(scored["ou_cal_logloss"].mean())
            summary["ece_ou25_cal"] = rel_ou_cal.ece()
            summary["totals_calib"] = totals_calib
    return {"summary": summary, "rel_1x2": rel_1x2, "rel_ou": rel_ou,
            "rel_ou_cal": rel_ou_cal if has_cal else None,
            "totals": pd.DataFrame(totals)}


# ---- reporting -----------------------------------------------------------
def print_report(out: dict) -> None:
    s = out["summary"]
    print("\n=== Calibration backtest ===")
    for k in ("test_window", "competitive_only", "refits", "n_scored", "n_skipped_unknown_team"):
        print(f"  {k:24s} {s[k]}")
    if s["n_scored"] == 0:
        print("  (no scorable matches)")
        return

    def skill(model, base):
        return f"{model:.4f}  (baseline {base:.4f}, skill {(base-model)/base*100:+.1f}%)"

    print("\n  1X2 (lower is better):")
    print(f"    log loss   {skill(s['model_logloss_1x2'], s['base_logloss_1x2'])}")
    print(f"    RPS        {skill(s['model_rps'], s['base_rps'])}")
    print(f"    Brier      {s['model_brier_1x2']:.4f}")
    print(f"    accuracy   {s['accuracy_1x2']*100:.1f}%")
    print(f"    ECE        {s['ece_1x2']:.4f}")
    print("\n  Over/Under 2.5 (lower is better):")
    print(f"    log loss   {skill(s['model_logloss_ou25'], s['base_logloss_ou25'])}")
    print(f"    ECE        {s['ece_ou25']:.4f}")
    if "ece_ou25_cal" in s:
        a, b = s["totals_calib"]
        print(f"    -- recalibrated (Platt a={a:+.3f}, b={b:.3f}):")
        print(f"    log loss   {skill(s['model_logloss_ou25_cal'], s['base_logloss_ou25'])}")
        print(f"    ECE        {s['ece_ou25_cal']:.4f}  "
              f"(was {s['ece_ou25']:.4f}, {(1-s['ece_ou25_cal']/s['ece_ou25'])*100:+.0f}%)")

    print("\n  1X2 reliability (pooled home/draw/away):")
    print(out["rel_1x2"].table().to_string(index=False,
          formatters={"mean_pred": "{:.3f}".format, "empirical": "{:.3f}".format}))
    print("\n  O/U 2.5 reliability (raw over prob):")
    print(out["rel_ou"].table().to_string(index=False,
          formatters={"mean_pred": "{:.3f}".format, "empirical": "{:.3f}".format}))
    if out.get("rel_ou_cal") is not None:
        print("\n  O/U 2.5 reliability (recalibrated):")
        print(out["rel_ou_cal"].table().to_string(index=False,
              formatters={"mean_pred": "{:.3f}".format, "empirical": "{:.3f}".format}))


def save_plot(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    def draw(ax, t, label, **kw):
        ax.plot(t["mean_pred"], t["empirical"], "o-", label=label, **kw)

    # panel 1: 1X2
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    draw(ax, out["rel_1x2"].table(), "model")
    ax.set_title(f"1X2 (pooled)  (ECE {out['rel_1x2'].ece():.3f})")

    # panel 2: O/U, raw and (if present) recalibrated
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    draw(ax, out["rel_ou"].table(), f"raw (ECE {out['rel_ou'].ece():.3f})", color="tab:red")
    if out.get("rel_ou_cal") is not None:
        draw(ax, out["rel_ou_cal"].table(),
             f"recalibrated (ECE {out['rel_ou_cal'].ece():.3f})", color="tab:green")
    ax.set_title("Over/Under 2.5")

    for ax in axes:
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("empirical frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
    fig.suptitle(f"Calibration  ({out['summary']['test_window']}, n={out['summary']['n_scored']})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nsaved reliability plot -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward calibration backtest")
    p.add_argument("--test-start", default="2022-06-01")
    p.add_argument("--test-end", default=str(date.today()))
    p.add_argument("--refit-days", type=int, default=180,
                   help="refit cadence; smaller = fresher but slower")
    p.add_argument("--competitive-only", action="store_true",
                   help="exclude friendlies from the test set")
    p.add_argument("--totals-calib", type=float, nargs=2, default=None,
                   metavar=("A", "B"),
                   help="Platt a b for O/U recalibration; also evaluates raw vs. "
                        "calibrated. Get fitted values from `python src/calibrate.py`.")
    p.add_argument("--half-life", type=float, default=730.0)
    p.add_argument("--min-matches", type=int, default=8)
    p.add_argument("--elo", action="store_true",
                   help="add the point-in-time Elo covariate to the goal rates")
    p.add_argument("--elo-prior", action="store_true",
                   help="shrink each team's net strength toward its Elo rating (prior)")
    p.add_argument("--elo-lambda", type=float, default=5.0,
                   help="strength of the Elo-prior shrinkage (default 5)")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    calib = tuple(args.totals_calib) if args.totals_calib else None
    out = run(args.test_start, args.test_end, args.refit_days, args.competitive_only,
              args.half_life, args.min_matches, calib, use_elo=args.elo,
              elo_prior=args.elo_prior, elo_prior_lambda=args.elo_lambda)
    print_report(out)
    if not args.no_plot and out["summary"]["n_scored"]:
        save_plot(out, OUT_DIR / "calibration.png")


if __name__ == "__main__":
    main()
