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
python src/predict.py day 2026-06-21                   # full predictions printout for one day's WC slate
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
    predict.py         # CLI (fit / day / match / fixtures / rankings / teams)
    backtest.py        # walk-forward, no-leakage calibration backtest (+ feature experiment flags)
    calibrate.py       # fit + validate the over/under recalibration (Platt)
    track.py           # forward prediction tracker: freeze / grade / report (accuracy record)
    challenge.py       # daily $25->$100 challenge: log / grade / report (-EV fun)
    slate.py           # live workflow: board / log / grade / report (CLV by tier, Asian handicap)
    odds.py            # The Odds API: auto value-scan + auto-logging (h2h/totals/spreads, tiered)
    kelly.py           # tier-weighted fractional-Kelly staking off the forward bet log
    elo.py             # [experiment, off] point-in-time World Football Elo from results
    squad.py           # [experiment, off] point-in-time squad market value (Transfermarkt)
    context.py         # [experiment, off] point-in-time rest / travel / climate covariates
    test_model.py      # hermetic sanity tests
  data/                # csvs + cached model.json (gitignored); committed: predictions.csv,
                       #   challenge_log.csv, country_coords.csv
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
not modelled here** — the Dixon-Coles model prices team goals, not individual players.
A separate anytime-goalscorer prototype exists (see *Player props* under Next steps).

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

## Daily predictions (the main use)

The everyday command is a clean terminal printout of one day's World Cup slate:

```bash
python src/download_data.py --force      # refresh results (overnight games)
python src/predict.py day 2026-06-21     # full predictions for that day
```

For each match `day` prints: kickoff time, the predicted scoreline + expected goals,
1X2 win/draw/win probabilities (for context) with fair odds, over/under 2.5, BTTS, and
the single most-likely outcome. The **headline confidence is Asian-handicap cover
confidence** — how confident the model is that the favourite covers the book's main
(most-balanced) handicap line, scored by how far the model's cover probability sits from
a 50/50 coin-flip. (Match-winner confidence is near-useless — "Spain beats Saudi Arabia"
is obvious; "Spain covers −2.5" is the real question.) It closes with a **biggest-
mismatches** ranking by handicap-cover disagreement (model cover vs the posted line).

The handicap line + de-vigged implied prob come from the same odds feed the betting
layer uses (~3 credits per run); `--no-odds` runs fully offline (no kickoff times, no AH
confidence). The model prices **team goals only** — no player props.

### Tracking accuracy (`track.py`)

To honestly compare predictions to outcomes over the tournament, `track.py` freezes a
**point-in-time** prediction for each fixture and grades it once the game is played:

```bash
python src/track.py freeze     # lock predictions for upcoming fixtures (next 3 days)
python src/track.py grade      # fill actual results + per-market correctness
python src/track.py report     # accuracy: 1X2 hit rate, exact score, O/U, BTTS, Brier, log loss
```

No leakage: only **unplayed** fixtures are frozen (an unplayed game can't be in the
training data), and each is recorded once — a later refit never rewrites a locked
prediction, so the record is a true forward forecast. The log lives at
`data/predictions.csv` (tracked in git — it *is* the accuracy record). The daily
`soccer-refresh` job runs `freeze` + `grade` automatically each morning.

### $25 → $100 challenge (`challenge.py`)

A daily for-fun exercise: pick a ticket of bets on tomorrow's slate and see whether $25
would have reached $100. Logged and graded (from live scores) so results accumulate.

```bash
python src/challenge.py log --date 2026-06-22 --stake 25 --legs '<json legs>'
python src/challenge.py grade    # settle from played games (--live pulls Odds API scores)
python src/challenge.py report   # per-day: did $25 reach $100?
```

It's **-EV gambling on the model's known bias, not financial advice** — a 4x target
rewards concentration (one small parlay), not spreading into many singles.

The betting machinery below is no longer the focus but stays usable. `scan --days 2`
ranks 48h of picks by EV with an inline `!LEAN:draw/dog/under` flag; `kelly.py` with no
settled sample prints a flat, capped **cold-start** probe on HIGH-tier picks (Kelly is $0
with zero evidence — a deliberate, labelled override to seed the CLV record).

```powershell
python src/download_data.py --force ; python src/slate.py grade   # refresh results + grade yesterday
python src/odds.py scan --days 2                                   # ranked 48h slate, tier + !LEAN flags
python src/odds.py log --min-edge 0.05                             # auto-log HIGH-tier picks at opening price
python src/kelly.py --unit 5 --bankroll 500                        # cold-start stake (set your own $unit/bankroll)
python src/odds.py close                                           # near kickoff: snapshot closing lines
python src/slate.py report                                         # by-tier CLV scoreboard
```

CLV is only honest if the closing snapshot is **near kickoff**. The scheduled jobs split
this correctly: **`soccer-refresh`** (9am ET → `scripts/daily_refresh.ps1`: refresh +
grade + report) and **`soccer-close`** (11:40/14:40/17:40/20:40 ET →
`scripts/daily_close.ps1`: `odds.py close`, which overwrites so the *last pre-kickoff run
wins* = true close). Manage via `Get-ScheduledTask soccer-*`.

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

### Model accuracy — what we tried, measured, and concluded

The baseline sees only team identity + venue (~25 effective weighted matches per team).
We hypothesised that adding signal would lift skill, and tested it **measurement-first**:
each candidate was built point-in-time (no leakage), added to the walk-forward backtest,
and judged purely by out-of-sample skill vs the baseline (**1X2 log loss 0.847, RPS 0.165,
60% accuracy, ECE 0.013** over 2,534 competitive matches, 2023-06→2026-06). All toggles
default **off**; the production model is the baseline.

| Experiment (flag) | 1X2 log loss | Verdict |
|---|---|---|
| baseline | **0.847** | — |
| Elo covariate (`--elo`, `elo.py`) | 0.848 | flat — Elo is a re-encoding of results the model already fits |
| Elo prior (`--elo-prior`) | 0.853 | worse |
| Squad-value prior (`--squad-prior`, `squad.py`) | 0.865 | worst — free Transfermarkt value is sparse/misaligned for the minnows it's meant to help |
| Context: rest/travel/climate (`--context`, `context.py`) | 0.849 | flat — effects are real but tiny (travel coef ≈6% of home-adv for a full-SD trip) |

**Conclusion: the model is at the accuracy ceiling its free data allows.** The failures
weren't an architecture problem (a tree/regression can't extract signal a feature doesn't
carry) — Elo/squad are redundant with results, and genuinely-independent context
(rest/travel) has only a tiny effect on international match outcomes (a low-scoring,
high-variance sport with large irreducible uncertainty). 60% accuracy / ECE 0.013 is
near the soccer-1X2 ceiling pro models also hit. The experiment modules are kept as
documented, off-by-default reference.

**The one lever not pursued** (ruled out as paid): clean per-match lineups/injuries
(e.g. API-Football, ~$19/mo) for an exact squad-quality feature — the only realistic path
to *new* information, and even that is an uncertain bet. Other ideas if ever revisited:
Poisson-lognormal in place of the DC ρ correction (a draw-*calibration* change, not an
accuracy one — the model already over-predicts draws); a market-anchored 1X2 layer
(a CLV project, not accuracy); per-team strength uncertainty for honest bet sizing.

### Player props — anytime goalscorer (prototype, not yet in `src/`)

The match-outcome market is provably efficient (a public-data DC model can't beat the WC
closing line — confirmed on 1,971 historical WC matches). **Player props are the better
target**: books post them soft, at low limits, across few books — room a model can
exploit. Prop markets *are* live and bettable on The Odds API for `soccer_fifa_world_cup`
(`player_goal_scorer_anytime`, `player_shots`, `player_shots_on_target` — 5 books,
~6 credits/event via the event-odds endpoint).

A working prototype lives in `c:\tmp` (not committed): it builds a per-player
anytime-goalscorer model and merges it against live book prices to surface edges.
Two data sources were tried:
- **StatsBomb open data** (free, GitHub — WC 2018/2022, Euro 2024) gives per-shot xG but
  only ~8 international appearances per player: far too thin to estimate a shot rate.
- **Club shot data** (FBref top-5 leagues 2025/26, via a Kaggle mirror) gives 20-35 club
  matches per player — stable rates. **This is the spine.** The model anchors each
  player's club goals/90 to the team's expected goals for *this* match (from the DC
  `lambdas`, which already encode opponent + venue), then Poisson → P(scores ≥1). The
  book's player list conveniently filters to the actual squad.

Result: the club-data model **tracks the market closely** on the players it covers
(30/46 priced players, most within a few pp) — a sane baseline, not the ±25pp noise the
international-only version produced. But the visible "edges" aren't bankable yet:
1. **Add xG** for shot quality (the FBref mirror lacks it; the stars get underrated). Pull
   an Understat-derived Kaggle set — Understat itself is now JS-gated, FBref 403s urllib,
   but the Kaggle bearer-token download works.
2. **Lineups/minutes** — the biggest error source is assuming who starts and for how long.
3. **Coverage beyond top-5** — real starters who left for Turkey/Saudi/MLS (Enner Valencia,
   Sané) drop out.
4. **Forward-tracking is the only validation** — historical prop *odds* aren't free, so
   log model picks vs actual goalscorers daily (mirror `track.py`) and let hit-rate/CLV
   accumulate. Do this regardless; it's the only thing that can prove or kill the edge.
