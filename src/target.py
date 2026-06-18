"""Target staking: 'turn $B into $T tonight' -> the optimal bet structure.

This is a ruin/target problem, not an edge problem. When you must reach a target
through a subfair game (the vig), Dubins-Savage 'bold play' is optimal: bet as
concentrated as possible to reach the target in one shot. Diversifying across many
small bets just grinds you toward the vig — fatal when you need an outlier.

So this searches the day's games for the **single parlay** (one leg per game, legs
independent across games) whose payout clears the target with the **highest model
probability of hitting**. Live prices come from The Odds API via odds.py.

    python src/target.py --bankroll 20 --target 100 --date 2026-06-18
    python src/target.py -b 20 -t 100 --max-legs 4

Honesty: it also prints the survival probability. For a 5x target that number is
usually grim — the tool's most useful output is often 'don't make this bet'.
"""
from __future__ import annotations

import argparse
import functools
import itertools
import operator
from datetime import date, timedelta
from types import SimpleNamespace

import odds


def _prod(xs) -> float:
    return functools.reduce(operator.mul, xs, 1.0)


def best_plans(games: list[list[dict]], bankroll: float, target: float,
               max_legs: int, max_leg_odds: float = 8.0):
    """Enumerate parlays (one selection per game) that reach the target; rank by
    model P(all legs hit). `max_leg_odds` skips wild longshot legs whose model
    probability is least trustworthy."""
    options = []
    for rows in games:
        opts = [r for r in rows if r["price"] and r["model_prob"] > 0
                and r["price"] <= max_leg_odds]
        options.append(opts)

    out = []
    n = len(options)
    for k in range(1, min(max_legs, n) + 1):
        for idxs in itertools.combinations(range(n), k):
            for combo in itertools.product(*[options[i] for i in idxs]):
                odds_mult = _prod(c["price"] for c in combo)
                if bankroll * odds_mult < target:
                    continue
                p = _prod(c["model_prob"] for c in combo)
                out.append({"p_hit": p, "odds": odds_mult,
                            "payout": bankroll * odds_mult, "legs": list(combo)})
    out.sort(key=lambda d: -d["p_hit"])
    return out


def _leg_str(c: dict) -> str:
    return (f"{c['home']} v {c['away']} -> {c['label']} "
            f"@ {c['price']} ({c['book']}, model {c['model_prob']*100:.0f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description="Target staking optimizer (bold play)")
    p.add_argument("-b", "--bankroll", type=float, default=20.0)
    p.add_argument("-t", "--target", type=float, default=100.0)
    p.add_argument("--date", default=str(date.today() + timedelta(days=1)),
                   help="restrict to one match date YYYY-MM-DD (default tomorrow)")
    p.add_argument("--max-legs", type=int, default=4)
    p.add_argument("--max-leg-odds", type=float, default=8.0,
                   help="skip longshot legs above this price (untrustworthy model probs)")
    p.add_argument("--regions", default="us")
    p.add_argument("--sport", default=None)
    args = p.parse_args()

    _, sport, events, matched, unmatched, hdr = odds._gather(
        SimpleNamespace(sport=args.sport, regions=args.regions))
    games = [rows for rows in matched if rows and rows[0]["date"] == args.date]
    if not games:
        print(f"No matched games on {args.date}. Matched dates available: "
              f"{sorted({r[0]['date'] for r in matched if r})}")
        return

    print(f"Target: ${args.bankroll:.0f} -> ${args.target:.0f}  "
          f"({args.target/args.bankroll:.1f}x) on {args.date}, {len(games)} games\n")

    plans = best_plans(games, args.bankroll, args.target, args.max_legs, args.max_leg_odds)
    if not plans:
        print("No parlay up to --max-legs reaches the target. Raise --max-legs or "
              "--max-leg-odds, or accept the target is out of reach at this bankroll.")
        return

    best = plans[0]
    print(f"BEST PLAY (bold): stake the full ${args.bankroll:.0f} on this "
          f"{len(best['legs'])}-leg parlay")
    for c in best["legs"]:
        print(f"    - {_leg_str(c)}")
    print(f"  combined odds {best['odds']:.2f}  ->  pays ${best['payout']:.0f}")
    print(f"  model P(all hit) = {best['p_hit']*100:.1f}%   "
          f"==> you survive about {best['p_hit']*100:.0f}% of the time\n")

    print("Other structures (fewer legs = surer but must pay enough; ranked by P):")
    seen = set()
    shown = 0
    for pl in plans:
        sig = len(pl["legs"])
        if sig in seen:
            continue
        seen.add(sig)
        legs = " + ".join(f"{c['home'][:3]}/{c['label']}" for c in pl["legs"])
        print(f"    {sig}-leg  P {pl['p_hit']*100:4.1f}%  odds {pl['odds']:5.2f}  "
              f"${pl['payout']:4.0f}  [{legs}]")
        shown += 1
        if shown >= 4:
            break

    mult = args.target / args.bankroll
    print(f"\nReality: even the best structure is ~{best['p_hit']*100:.0f}% to hit. "
          f"A {mult:.0f}x target through the vig is mostly a coin you will lose -- the "
          f"honest recommendation is usually don't take the bet.")
    print(f"quota: {hdr['remaining']} remaining, {hdr['used']} used")


if __name__ == "__main__":
    main()
