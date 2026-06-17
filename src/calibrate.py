"""Fit the over/under (totals) recalibration out of sample, then validate it.

The backtest showed the raw Poisson over/under probabilities are *overconfident*:
they sit too far from the base rate at both tails (predict 0.95 -> ~0.76 happen;
predict 0.17 -> ~0.37 happen). This is not count overdispersion — conditional on
the fitted mu, totals are ~Poisson. It is a threshold-probability calibration
problem, so the fix is Platt scaling on the over-probability itself:

    p_cal = sigmoid(a + b * logit(p_raw))

with (a, b) fit by minimizing binary log loss. b < 1 compresses predictions
toward the base rate (curing overconfidence); a sets the base rate.

Must be fit out of sample (mu-hat carries realistic error there) and on a window
disjoint from the one used to grade it:
  1. CALIB window: walk-forward -> collect (raw_over, over_occurred) -> fit (a, b).
  2. Write (a, b) into the cached model.json (so the CLI uses calibrated O/U).
  3. TEST window (disjoint, later): score raw vs. calibrated O/U, report ECE
     before/after, save data/calibration.png.

Usage:
    python src/calibrate.py
    python src/calibrate.py --calib-start 2014-01-01 --calib-end 2022-06-01 \
                            --test-start 2022-06-01 --refit-days 240
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from dixon_coles import _logit, _sigmoid
import backtest

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.json"


def fit_platt(raw_over: np.ndarray, occurred: np.ndarray) -> tuple[float, float]:
    """(a, b) minimizing binary log loss of sigmoid(a + b*logit(raw_over))."""
    z = _logit(raw_over)
    y = occurred.astype(float)

    def bce(params):
        a, b = params
        p = np.clip(_sigmoid(a + b * z), 1e-12, 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

    res = minimize(bce, x0=np.array([0.0, 1.0]), method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def fit_calibration(calib_start: str, calib_end: str, refit_days: int,
                    half_life: float, min_matches: int) -> tuple[float, float]:
    print(f"[1/3] generating out-of-sample predictions over calibration window "
          f"{calib_start} -> {calib_end} ...")
    out = backtest.run(calib_start, calib_end, refit_days, competitive_only=True,
                       half_life=half_life, min_matches=min_matches)
    td = out["totals"]
    if td.empty:
        raise SystemExit("No predictions in calibration window.")
    a, b = fit_platt(td["raw_over"].to_numpy(), td["over_occurred"].to_numpy())
    print(f"      fit on {len(td):,} OOS matches: Platt a = {a:+.3f}, b = {b:.3f}  "
          f"(b<1 => shrink toward base rate)")
    return a, b


def write_to_model(calib: tuple[float, float]) -> None:
    if not MODEL_PATH.exists():
        print(f"[2/3] no cached model at {MODEL_PATH}; skipping write "
              f"(run `python src/predict.py fit`). totals_calib = {calib}")
        return
    d = json.loads(MODEL_PATH.read_text())
    d["totals_calib"] = list(calib)
    MODEL_PATH.write_text(json.dumps(d, indent=2))
    print(f"[2/3] wrote totals_calib = [{calib[0]:+.3f}, {calib[1]:.3f}] "
          f"into {MODEL_PATH.name}")


def validate(calib: tuple[float, float], test_start: str, test_end: str,
             refit_days: int, half_life: float, min_matches: int, plot: bool) -> None:
    print(f"[3/3] validating on disjoint test window {test_start} -> {test_end} ...")
    out = backtest.run(test_start, test_end, refit_days, competitive_only=True,
                       half_life=half_life, min_matches=min_matches, totals_calib=calib)
    backtest.print_report(out)
    if plot and out["summary"]["n_scored"]:
        backtest.save_plot(out, backtest.OUT_DIR / "calibration.png")


def main() -> None:
    p = argparse.ArgumentParser(description="Fit & validate O/U recalibration")
    p.add_argument("--calib-start", default="2014-01-01")
    p.add_argument("--calib-end", default="2022-06-01")
    p.add_argument("--test-start", default="2022-06-01")
    p.add_argument("--test-end", default=str(date.today()))
    p.add_argument("--refit-days", type=int, default=240)
    p.add_argument("--half-life", type=float, default=730.0)
    p.add_argument("--min-matches", type=int, default=8)
    p.add_argument("--no-write", action="store_true", help="don't touch model.json")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    calib = fit_calibration(args.calib_start, args.calib_end, args.refit_days,
                            args.half_life, args.min_matches)
    if not args.no_write:
        write_to_model(calib)
    validate(calib, args.test_start, args.test_end, args.refit_days,
             args.half_life, args.min_matches, plot=not args.no_plot)


if __name__ == "__main__":
    main()
