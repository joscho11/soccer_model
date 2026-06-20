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
    slate.py           # live workflow: board / log / grade / report (CLV by tier, Asian handicap)
    odds.py            # The Odds API: auto value-scan + auto-logging (h2h/totals/spreads, tiered)
    kelly.py           # tier-weighted fractional-Kelly staking off the forward bet log
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

## Live workflow (`slate.py`)

The model is calibrated; the missing piece is the *market*. During a live
tournament `slate.py` turns predictions into a daily loop and — critically — logs
every bet's `(model_prob, odds, result)` so that by the end you can measure edge
**against the line**, not just against base rates. Flat 1-unit stakes; log lives
at `data/bet_log.csv` (gitignored).

```bash
python src/slate.py board                                  # today+tomorrow fixtures, probs & fair odds
python src/slate.py log England Croatia --market home --odds 2.10
python src/slate.py log Portugal "DR Congo" --market over --line 2.5 --odds 2.05
python src/slate.py log Germany "Ivory Coast" --market ah_home --line -1.5 --odds 2.20  # Asian handicap
python src/slate.py close England Croatia --market home --odds 1.80          # closing line, for CLV
python src/slate.py close Germany "Ivory Coast" --market ah_home --line -1.5 --odds 2.05
python src/slate.py grade                                   # settle bets from played results
python src/slate.py report                                  # record, ROI, model-check, CLV by tier
```

**Markets** (all derived from the one score matrix): `home/draw/away` (1X2),
`over/under` (totals), `btts_yes/btts_no`, and **`ah_home/ah_away`** (Asian
handicap — the goal-supremacy "spread"; pass the signed handicap as `--line`,
e.g. `ah_home --line -1.5`). Whole/half lines only; quarter-lines aren't priced yet.

**Confidence tiers** (the soccer analog of the NFL model's HIGH/MEDIUM/PASS): every
bet is tagged by its model-vs-market edge — **HIGH ≥ 5pp, MEDIUM ≥ 3pp, PASS** below
(`slate.tier_for`). Auto-scanned bets tier on the edge vs the *de-vigged consensus*;
manually-logged bets can only see one price, so their tier is price-implied (labelled
as such).

`report` shows realised ROI, whether the model's EV was predictive (avg model
prob vs. realised win rate), and — where you've logged closing odds — **closing
line value by tier**: per HIGH/MEDIUM/PASS, the hit rate, ROI, and how often /
by how much you beat the close. This is the honest test: *does the HIGH tier
actually clear the close?* Positive CLV is the only signal that survives noisy
short-run results — exactly the lesson the sibling NFL model learned the hard way
(its "beats the close" edge turned out to be a closing-line-anchored artifact).

### Staking (`kelly.py`)

Once a tier has a settled forward sample, `kelly.py` sizes stakes by **fractional
Kelly off the Wilson 95% lower bound** of that tier's hit rate (not the point
estimate), capped per bet — so a lucky small sample can't inflate stakes, and a
tier whose conservative edge doesn't clear the price gets **$0**.

```bash
python src/kelly.py --kelly-fraction 0.25 --cap 0.02 --min-n 20
```

A positive hit rate is necessary but **not sufficient** — confirm positive
per-tier CLV (above) before trusting any stake. Against the sharp WC market the
honest prior is that no tier has real edge until the forward CLV proves it.

## Automated odds (`odds.py`)

Rather than typing prices by hand, `odds.py` pulls live US-book odds (FanDuel,
DraftKings, BetMGM, …) from **[The Odds API](https://the-odds-api.com)** — a
sanctioned JSON feed, *not* scraping — takes the best available price per market
across books, and runs the value scan automatically. Stdlib only, no extra deps.

The free tier is ~500 credits/month (cost = markets × regions, so an h2h+totals
scan on `us` is 2 credits ≈ 250 scans/month — ample for one tournament). Get a
free key (no card), then:

```bash
export ODDS_API_KEY=...                  # PowerShell: $env:ODDS_API_KEY="..."  (read from .env too)
python src/odds.py sports                 # find the World Cup sport key + see quota
python src/odds.py scan --min-odds 1.67   # value bets >= -150, sorted by EV, +odds flagged
python src/odds.py btts --date 2026-06-18 # both-teams-to-score value (per-event, 1 credit/game)
python src/odds.py log  --min-edge 0.03   # auto-log the +EV bets into bet_log.csv
```

`scan` shows American + decimal odds, the **confidence tier** (HIGH/MEDIUM/PASS,
see slate above), and `--min-odds` enforces an upside floor (1.67 = -150). h2h +
totals + **spreads (Asian handicap)** come from the cheap bulk endpoint (**3
credits/scan**); each AH line is priced from the score matrix and de-vigged against
its paired opposite side. BTTS and player props are event-level only — `btts`
fetches them per game (1 credit each). **Player props (goalscorer/shots/cards) are
not modelled** — the Dixon-Coles model prices team goals, not individual players (a
future build off `goalscorers.csv`).

`scan`/`log` auto-discover the World Cup sport key, request **pre-match** games
only (in-play prices are noise), match each game to the model (alias map + fuzzy
fallback; unmatched games are reported, never silently dropped), and dedupe so
re-running doesn't double-log. From there `grade`/`report` work as above.

**Edge is measured vs the de-vigged market consensus, not the best line.** Taking
the single most generous price across ~10 books (including soft offshore ones)
manufactures fake "value"; the honest benchmark is the vig-free consensus, and a
bet is only flagged when the model genuinely disagrees *and* a real price clears
it.

**Reality check (important):** against the sharp World Cup market the model shows
a *systematic* pro-draw / pro-underdog lean — it disagrees in one consistent
direction on almost every game. That is the classic Dixon-Coles weakness (Poisson
structurally inflates draws and compresses talent gaps) meeting the sharpest
soccer market there is. When a model disagrees with an efficient market that
consistently, the prior is that the **model is biased, not that the market is
wrong** — so treat scan flags as *hypotheses to test via CLV*, not green lights.
Log a small flat-stake sample, record closing lines, and only trust the lean if
it produces positive closing-line value over a real sample. Making the model
actually competitive here means recalibrating its 1X2 (draw) probabilities
against the market — a real project, not done.

## Caveats

- The World Cup market is sharp; treat model edges as a screen, not a guarantee.
  The honest goal is **calibration vs. posted odds**, not raw accuracy.
- International data is thin per team (~10 matches/year) and roster-volatile.
  Strengths are point estimates with real uncertainty; the `min_matches` filter
  drops teams with too little recent data rather than fitting them badly.
- `--odds-*` value flags expect **decimal** odds and assume the posted price is
  the fair line plus margin; without a live odds feed you enter prices by hand.

## Next steps

The strategy mirrors the sibling NFL model (`BettingEdgeContinued/`): predict the
score → derive every market → bet where the model disagrees with the book → tier by
confidence → size by Kelly → **judge everything by closing-line value, not raw
accuracy or climatology**. Two tracks, measurement-first.

### Betting execution (this WC — already scaffolded, finish + run)

- **Quarter-line Asian handicap.** US books hang lots of `.25/.75` lines; the scanner
  currently skips them (`markets.asian_handicap` is whole/half only). Add the standard
  two-line stake split (`+1.25` = ½ `+1.0` + ½ `+1.5`) in both pricing and grading.
- **Forward CLV collection.** No cheap historical international odds exist, so the CLV
  rig is forward-only: log every qualifying pick at the *opening* line, snapshot the
  *closing* line near kickoff (`slate close`), and accumulate `report`'s by-tier CLV
  over the tournament. Bet paper / micro-stakes until a tier shows **positive CLV**.
- **Gate Kelly on proven CLV.** `kelly.py` already sizes off the Wilson lower bound;
  only turn on real stakes for a tier once its forward CLV is positive over a real
  sample (the NFL re-fit landed at HIGH 2% / MEDIUM $0 — expect this market to be
  harder, given the documented pro-draw/pro-underdog lean).
- Auto-snapshot closing lines (extend `odds.py` to write `closing_odds` near kickoff),
  and record book *price* alongside lines so CLV can be measured in cents, not just
  points (a finding carried over from the NFL council review).

### Model accuracy (the longer arc — from the design-council review)

The current model sees only team identity + venue (~25 effective weighted matches per
team), which is why it shrinks strong teams to the mean and leans pro-draw/underdog.
The highest-lift additions, in order:

1. **A real team-strength rating** — SPI-style separate offensive/defensive ratings off
   *xG-adjusted* goals (plus Elo from eloratings.net, a free high-lift scrape). This is
   the single biggest accuracy gain.
2. **A squad-strength prior** — fixes thin data *and* roster volatility at once: anchor
   each team's attack/defence to a squad-quality index (Transfermarkt market value, or
   league-strength-adjusted club xG of the projected XI) so a friendly B-team result
   stops contaminating the WC-XI estimate.
3. **Keep Dixon-Coles as the coherent score-matrix core**, but feed it better rates —
   either a LightGBM predicting the log-residual over the DC baseline, or a full
   Bayesian hierarchical dynamic DC (random-walk team states, partial pooling by
   confederation, squad covariate as the prior mean). Consider replacing the DC ρ
   correction with a Poisson-lognormal to cure the structural draw bias.
4. **Per-team strength uncertainty** (bootstrap/Bayesian posterior) to size bets honestly.
5. **Data engineering caveat (Jan 2026):** the free FBref/Opta xG feed was shut off —
   xG now comes from StatsBomb Open Data (tournament snapshots) or a paid feed
   (API-Football has explicit WC 2026 coverage incl. lineups/injuries). Build a
   point-in-time feature store with archived raw snapshots (no leakage).
6. Re-fit `calibrate.py` as the live record grows; swap `backtest.py`'s climatology
   baseline for a true market baseline once a historical odds source is in place.
