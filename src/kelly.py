"""Tier-weighted fractional-Kelly staking, sized off the forward bet log.

Mirrors the NFL BettingEdge kelly_staking.py, but international football has no
cheap historical closing-odds set to backtest against — so the honest win rate
comes from the *forward* record in data/bet_log.csv (graded by slate.py), not a
walk-forward backtest. Until a tier has a real settled sample, it gets $0.

Sizing, conservatively (same discipline as the NFL module):
  1. Win prob = each tier's settled hit rate, taken at the Wilson 95% lower bound
     (not the point estimate) so a lucky small sample can't inflate stakes.
  2. Fractional Kelly (default 1/4) at the tier's typical price + a hard per-bet cap.
  3. Tiers whose conservative edge doesn't clear the break-even price get $0.

    python src/kelly.py                         # staking plan from the current log
    python src/kelly.py --kelly-fraction 0.25 --cap 0.02 --min-n 20

CAVEAT: a positive *hit rate* is not proof of edge — against a sharp market it can
be break-even or worse after vig, and short samples are noise. The signal to trust
is positive **closing-line value** (see `slate.py report` by-tier CLV). Treat this
plan as "what to stake IF the tier's edge is real", gated on that CLV holding up.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import slate


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a binomial rate."""
    if n == 0:
        return 0.0
    phat = wins / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (center - margin) / denom


def kelly_fraction(p: float, dec_odds: float) -> float:
    """Full-Kelly stake fraction for win prob p at decimal odds. Floored at 0."""
    b = dec_odds - 1
    if b <= 0:
        return 0.0
    return max(0.0, (p * (b + 1) - 1) / b)


def tier_stats(df: pd.DataFrame) -> dict:
    """{tier: (wins, n, median_decimal_odds)} from settled, decisive bets
    (pushes excluded) — the forward record to size Kelly against."""
    settled = df[df["result"].isin(["win", "loss"])].copy()  # decisive only (no push)
    settled["odds"] = pd.to_numeric(settled["odds"], errors="coerce")
    out = {}
    for tier, g in settled.groupby(settled["tier"].fillna("").replace("", "(none)")):
        wins = int((g["result"] == "win").sum())
        out[tier] = (wins, len(g), float(g["odds"].median()))
    return out


def cold_start_plan(df: pd.DataFrame, unit: float, bankroll: float,
                    cap_frac: float, min_n: int, days: int | None = None) -> None:
    """Staking for RIGHT NOW, before any tier has a settled forward sample.

    With n=0 settled bets the honest win-rate estimate is undefined, its Wilson
    95% lower bound is 0, and Kelly therefore says **$0** — betting nothing is the
    statistically correct move. But the only way to *acquire* the CLV evidence that
    would justify a real stake is to bet forward during the live tournament. So we
    deliberately override Kelly with a small, flat, capped *probe* on HIGH-tier
    picks only, explicitly speculative-until-CLV-proven, and print the math showing
    why it stays tiny and never scales off the model's unproven self-belief."""
    pending = df[df["result"] == "pending"].copy()
    high = pending[pending["tier"] == "HIGH"]
    if days:
        from datetime import date, timedelta
        window = {str(date.today() + timedelta(days=i)) for i in range(days)}
        high = high[high["date"].astype(str).isin(window)]
    settled_high = df[(df["tier"] == "HIGH") & df["result"].isin(["win", "loss"])]
    n_high = len(high)

    win_txt = f" in the next {days}d" if days else ""
    print(f"\n=== COLD-START staking{win_txt} (no settled sample yet) ===")
    print(f"  HIGH-tier settled bets: {len(settled_high)}  ->  Wilson 95% lower bound "
          f"of win rate = 0  ->  Kelly says $0.")
    print(f"  Kelly is correct that the *proven* edge is zero. We override with a flat,")
    print(f"  capped probe purely to start the forward CLV record. Speculative until CLV-proven.\n")

    if n_high == 0:
        print("  No pending HIGH-tier picks in the log. Run `odds.py log --min-edge 0.05` "
              "(or `slate log`)\n  to add some, then re-run.")
        return

    odds = pd.to_numeric(high["odds"], errors="coerce")
    probs = pd.to_numeric(high["model_prob"], errors="coerce")
    med_odds, med_prob = float(odds.median()), float(probs.median())

    # What Kelly WOULD stake if we naively trusted the model's own edge — shown only
    # to demonstrate how much we're choosing NOT to bet while the edge is unproven.
    f_naive = kelly_fraction(med_prob, med_odds)
    naive_pct = f_naive * 0.25  # quarter-Kelly
    naive_dollars = naive_pct * bankroll

    flat_pct = unit / bankroll if bankroll else 0.0
    total = unit * n_high
    total_pct = total / bankroll if bankroll else 0.0

    print(f"  pending HIGH picks      {n_high}")
    print(f"  flat probe / pick       ${unit:.2f}  ({flat_pct*100:.2f}% of ${bankroll:,.0f} bankroll)")
    print(f"  total speculative risk  ${total:.2f}  ({total_pct*100:.2f}% of bankroll)")
    cap_dollars = cap_frac * bankroll
    if total > cap_dollars:
        scaled = cap_dollars / n_high
        print(f"  !! exceeds the {cap_frac*100:.0f}% cap (${cap_dollars:.0f}); "
              f"scale each pick to ${scaled:.2f} (or bet fewer).")
    else:
        print(f"  within the {cap_frac*100:.0f}% total-exposure cap (${cap_dollars:.0f}). OK.")

    print(f"\n  why the stake stays flat & small (the guardrail):")
    print(f"    - the model believes its median HIGH pick is {med_prob*100:.0f}% @ {med_odds:.2f}; "
          f"taken at\n      face value that's quarter-Kelly ~{naive_pct*100:.1f}% "
          f"(${naive_dollars:.0f}/bet) -- we IGNORE that.")
    print(f"    - against a sharp market a one-directional disagreement is most likely the")
    print(f"      MODEL's bias (see !LEAN picks), so the prior edge is ~0 -> flat ${unit:.2f}, not Kelly.")
    print(f"    - scale a tier ONLY when its forward CLV turns positive (slate.py report).")


def main() -> None:
    p = argparse.ArgumentParser(description="Tier-weighted fractional-Kelly staking plan")
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--kelly-fraction", type=float, default=0.25, help="fraction of full Kelly (default 1/4)")
    p.add_argument("--cap", type=float, default=0.02, help="max stake per bet as bankroll fraction (default 2%)")
    p.add_argument("--confidence", type=float, default=1.96, help="z for the Wilson lower bound (default 95%)")
    p.add_argument("--min-n", type=int, default=20, help="min settled bets before a tier is sized (default 20)")
    p.add_argument("--unit", type=float, default=5.0,
                   help="cold-start flat probe $ per HIGH-tier pick (set your own amount)")
    p.add_argument("--cold-start-cap", type=float, default=0.05,
                   help="max TOTAL speculative exposure as bankroll fraction (default 5%)")
    p.add_argument("--days", type=int, default=2,
                   help="cold-start: only count pending HIGH picks in this window (default 2; 0 = all)")
    args = p.parse_args()

    df = slate._read_log()
    stats = tier_stats(df)
    if not stats or all(n < args.min_n for _, n, _ in stats.values()):
        # nothing has a sized sample yet -> cold-start is the operative plan
        cold_start_plan(df, args.unit, args.bankroll, args.cold_start_cap,
                        args.min_n, days=args.days)
        return

    print("=== Tier-weighted Kelly staking plan (from forward bet log) ===")
    print(f"  bankroll ${args.bankroll:,.0f}   sizing: {args.kelly_fraction:g} Kelly off the "
          f"Wilson {args.confidence == 1.96 and '95%' or ''} lower bound, capped at {args.cap*100:.1f}%/bet")
    print(f"  (a tier needs >= {args.min_n} settled bets to be sized)\n")
    print(f"  {'tier':8s} {'n':>4} {'hit%':>6} {'95%lo':>7} {'medOdds':>8} "
          f"{'b/e%':>6} {'fullKelly':>10} {'stake%':>7} {'$/bet':>7}")
    print("  " + "-" * 74)

    plan = []
    # only HIGH/MEDIUM are bettable tiers; PASS = "don't bet" by construction.
    for tier in ["HIGH", "MEDIUM", "PASS", "(none)"]:
        if tier not in stats:
            continue
        wins, n, med_odds = stats[tier]
        phat = wins / n if n else 0.0
        breakeven = 1 / med_odds if med_odds and med_odds > 1 else float("nan")
        if n < args.min_n or tier in ("PASS", "(none)"):
            note = "  <- not sized " + ("(insufficient n)" if n < args.min_n else "(non-bet tier)")
            print(f"  {tier:8s} {n:>4} {phat*100:5.0f}% {'—':>7} {med_odds:8.2f} "
                  f"{breakeven*100:5.0f}% {'—':>10} {'0.0%':>7} {0:6.0f}{note}")
            continue
        p_lo = wilson_lower(wins, n, args.confidence)
        f_full = kelly_fraction(p_lo, med_odds)
        stake_pct = min(f_full * args.kelly_fraction, args.cap) if f_full > 0 else 0.0
        dollars = stake_pct * args.bankroll
        flag = "" if stake_pct > 0 else "  <- no edge at this price"
        print(f"  {tier:8s} {n:>4} {phat*100:5.0f}% {p_lo*100:6.1f}% {med_odds:8.2f} "
              f"{breakeven*100:5.0f}% {f_full*100:9.1f}% {stake_pct*100:6.1f}% {dollars:6.0f}{flag}")
        if stake_pct > 0:
            plan.append((tier, stake_pct, dollars))

    print()
    if plan:
        for t, pct, d in plan:
            print(f"  => {t}: stake {pct*100:.1f}% of bankroll (${d:.0f}) per play.")
    else:
        print("  => No tier clears break-even on the conservative read yet.")
    print("\n  Reminder: hit rate clears the vig is necessary but NOT sufficient — confirm "
          "positive\n  closing-line value per tier (slate.py report) before trusting these stakes.")


if __name__ == "__main__":
    main()
