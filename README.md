# International Soccer Model

A Dixon-Coles goals model for international football, built to price World Cup
betting markets. One fitted rate model generates every market — 1X2, over/under,
BTTS, Asian handicap, correct score — from a single score matrix, and compares
those probabilities to posted odds to flag value.

## Approach

Each team has an **attack** and **defence** strength; a global **home-advantage**
term applies only at non-neutral venues (most World Cup games are neutral); a
low-score dependence parameter **rho** corrects the independent-Poisson
assumption (Dixon & Coles, 1997).

Fitting is weighted maximum likelihood:
- **Exponential time decay** (default 2-year half-life) so recent form dominates.
- **Competition-importance weights** so friendlies count roughly half a World Cup
  match — international friendlies are noisy (experimental line-ups, low stakes).

Data: [martj42/international_results](https://github.com/martj42/international_results)
— ~49k internationals from 1872 to present, with `neutral` and `tournament`
columns the model leans on. The repo also ships the upcoming WC 2026 fixtures.

## Setup

```bash
pip install -r requirements.txt
python src/download_data.py        # pull results.csv etc. into data/
python src/predict.py fit          # fit + cache the model (~35s)
python src/calibrate.py            # fit the over/under recalibration into model.json (~10min)
```

`fit` overwrites `model.json` and clears the recalibration, so re-run
`calibrate.py` after any refit (it appends the totals_calib params back).

## Usage

```bash
python src/predict.py rankings                         # team strength table
python src/predict.py match "Brazil" "Croatia"         # full market report (neutral venue)
python src/predict.py match "United States" Wales --home-venue   # host plays at home
python src/predict.py match France Senegal --odds-home 1.95      # value vs a posted price
python src/predict.py fixtures --limit 20              # predict upcoming WC 2026 games
python src/predict.py teams Korea                      # search available team names
```

Example:

```
Brazil vs Croatia  (neutral)
expected goals: Brazil 1.48 - 0.95 Croatia
  1X2     Brazil  48.7% (fair 2.05)   Draw  27.7% (fair 3.61)   Croatia  23.6% (fair 4.23)
  O/U 2.5 over  44.0%   under  56.0%
  BTTS    yes  48.2%   no  51.8%
```

## Layout

```
soccer_model/
  src/
    download_data.py   # pull the dataset
    dixon_coles.py     # the model: fit (weighted MLE) + score matrix
    markets.py         # score matrix -> market probabilities + value-vs-odds
    predict.py         # CLI (fit / match / fixtures / rankings / teams)
    backtest.py        # walk-forward, no-leakage calibration backtest
    calibrate.py       # fit + validate the over/under recalibration (Platt)
    test_model.py      # hermetic sanity tests
  data/                # downloaded csvs + cached model.json (gitignore-able)
```

## Calibration backtest

Before trusting any edge, `src/backtest.py` runs a **walk-forward, no-leakage**
evaluation: a match on date *d* is only ever scored by a model refit strictly
before *d*. Predictions are scored with proper scoring rules (log loss, RPS,
Brier) against a **climatology baseline** (venue-split historical base rates),
plus reliability curves and ECE. Saves `data/calibration.png`.

```bash
python src/backtest.py --test-start 2022-06-01 --refit-days 180 --competitive-only
```

**Results (2022-06-01 → 2026-06-17, 3,108 competitive matches):**

| market | model log loss | baseline | skill | ECE |
|--------|---------------:|---------:|------:|----:|
| 1X2          | 0.865 | 1.055 | **+18.0%** | **0.013** |
| O/U 2.5 raw  | 0.688 | 0.697 | +1.2% | 0.063 |
| O/U 2.5 *recalibrated* | 0.670 | 0.697 | **+3.8%** | **0.021** |

The **1X2 (match outcome) model is skillful and well-calibrated** — the
reliability curve hugs the diagonal across every probability bin, accuracy 60%.
Trust these probabilities for moneyline / double-chance / handicap pricing.

The **raw totals model was overconfident** — it *under*-predicted overs in
low-total games and *over*-predicted them in high-total games (ECE ~5× worse than
1X2). Investigation showed this is **not** count overdispersion (conditional on
the fitted μ, totals are ~Poisson) but out-of-sample error in μ̂: extreme
predicted totals regress to the mean. The fix is the totals recalibration below.

## Totals recalibration

`src/calibrate.py` fixes the over/under overconfidence with **Platt scaling** on
the over-probability, `p_cal = sigmoid(a + b·logit(p_raw))`, fit out of sample
(where μ̂ carries realistic error) on a window disjoint from the one used to grade
it. `b < 1` compresses predictions toward the base rate.

```bash
python src/calibrate.py            # fit on 2014–2022, write to model.json, validate on 2022–2026
```

Fitted **a ≈ 0.00, b ≈ 0.55** (the raw totals model is ~2× overconfident in logit
space; the base rate was already right). Validated out of sample, it cuts O/U
**ECE from 0.063 → 0.021 (−67%)** and roughly triples O/U skill over climatology
(+1.2% → +3.8%), leaving 1X2 untouched. `match` and `backtest` use it once the
parameters are written to `model.json`. Even calibrated, **O/U barely beats
climatology** — international totals are a near-efficient market — so calibration
here is about honest staking probabilities, not a goldmine.

## Caveats

- The World Cup market is sharp; treat model edges as a screen, not a guarantee.
  The honest goal is **calibration vs. posted odds**, not raw accuracy.
- International data is thin per team (~10 matches/year) and roster-volatile.
  Strengths are point estimates with real uncertainty; the `min_matches` filter
  drops teams with too little recent data rather than fitting them badly.
- `--odds-*` value flags expect **decimal** odds and assume the posted price is
  the fair line plus margin; without a live odds feed you enter prices by hand.

## Next steps

- Ingest a live odds feed to automate the value scan across all fixtures — then
  swap the climatology baseline in `backtest.py` for a true market baseline.
- Add per-team strength uncertainty (bootstrap or Bayesian) to size bets.
- Re-fit `calibrate.py` periodically as the live record grows; consider a
  separate Platt map for O/U 3.5 if you bet that line.
