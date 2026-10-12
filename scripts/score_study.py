"""Unblind a finished study and test each comparison against chance.

Every comparison is a two-alternative forced choice, so under the null of "no
perceptible difference" each pick is a coin flip. The exact binomial test is
computed directly, and the Wilson interval is used for the proportion because
it behaves near 0 and 1 where the normal approximation does not.

Usage:
    python -m scripts.score_study
    python -m scripts.score_study --results ~/Downloads/results.json
"""

import argparse
import json
import math
from pathlib import Path

from src.config import CLASS_NAMES, STUDY_DIR

COMPARISONS = {
    "ab": ("A", "B", "A vs B"),
    "real_vs_a": ("real", "A", "real vs A"),
    "real_vs_b": ("real", "B", "real vs B"),
}


def binomial_p(wins: int, n: int) -> float:
    """Two-sided exact binomial test against p = 0.5."""
    if n == 0:
        return 1.0
    observed = math.comb(n, wins)
    total = sum(math.comb(n, k) for k in range(n + 1) if math.comb(n, k) <= observed)
    return min(1.0, total / 2**n)


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def detectable(n: int) -> float:
    """Smallest win rate that would reach p < 0.05 at this sample size."""
    for wins in range(n // 2, n + 1):
        if binomial_p(wins, n) < 0.05:
            return wins / n
    return 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", type=Path, default=STUDY_DIR)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--by-class", action="store_true")
    args = parser.parse_args()

    key = json.loads((args.study / "key.json").read_text())
    results_path = args.results or args.study / "results.json"
    if not results_path.exists():
        raise SystemExit(f"No results at {results_path}. Download them from the page first.")
    results = json.loads(results_path.read_text())

    if results.get("created") != key.get("created"):
        raise SystemExit("Results do not match this study; they were rated against a different build.")

    print(f"A = {key['models']['A']}")
    print(f"B = {key['models']['B']}")
    print(f"guidance {key['guidance']}, {key['steps']} DDIM steps\n")

    tallies = {name: {} for name in COMPARISONS}
    per_class = {name: {} for name in COMPARISONS}
    skipped = 0

    for trial_id, choice in results["ratings"].items():
        if choice == "skip":
            skipped += 1
            continue
        trial = key["trials"][trial_id]
        winner = trial[choice]
        kind = trial["type"]
        tallies[kind][winner] = tallies[kind].get(winner, 0) + 1
        bucket = per_class[kind].setdefault(trial["class_id"], {})
        bucket[winner] = bucket.get(winner, 0) + 1

    for kind, (first, second, label) in COMPARISONS.items():
        wins_first = tallies[kind].get(first, 0)
        wins_second = tallies[kind].get(second, 0)
        n = wins_first + wins_second
        if n == 0:
            continue

        rate = wins_first / n
        low, high = wilson(wins_first, n)
        p = binomial_p(max(wins_first, wins_second), n)
        verdict = "significant" if p < 0.05 else f"not significant (need {detectable(n):.0%})"

        print(f"{label}")
        print(f"  {first} {wins_first} - {wins_second} {second}   (n = {n})")
        print(f"  {first} preferred {rate:.1%}  95% CI [{low:.1%}, {high:.1%}]")
        print(f"  p = {p:.4f}  {verdict}\n")

    if args.by_class:
        print("Per class (descriptive only, too few trials each to test):")
        for kind, (first, second, label) in COMPARISONS.items():
            if not per_class[kind]:
                continue
            print(f"  {label}")
            for class_id in sorted(per_class[kind]):
                bucket = per_class[kind][class_id]
                print(f"    {CLASS_NAMES[class_id]:<26} {first} {bucket.get(first, 0)} - {bucket.get(second, 0)} {second}")
            print()

    rated = len(results["ratings"]) - skipped
    print(f"{rated} trials rated" + (f", {skipped} skipped" if skipped else ""))


if __name__ == "__main__":
    main()
