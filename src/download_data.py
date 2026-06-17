"""Download the martj42 international-football-results dataset.

Source: https://github.com/martj42/international_results
~47k international matches from 1872 to present, with a `neutral` flag and
`tournament` label that the model relies on.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlopen

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ["results.csv", "shootouts.csv", "goalscorers.csv"]


def download(force: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = DATA_DIR / name
        if dest.exists() and not force:
            print(f"skip {name} (exists, {dest.stat().st_size:,} bytes)")
            continue
        url = f"{BASE}/{name}"
        print(f"downloading {url}")
        with urlopen(url) as r:
            dest.write_bytes(r.read())
        print(f"  saved {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    download(force="--force" in sys.argv)
