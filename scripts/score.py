#!/usr/bin/env python3
"""
score.py -- unblind the judgement and apply the sealed criterion.

There is nothing to decide here. The thresholds were fixed in the
preregistration before any model was trained; this script only counts.

Criterion (two-sided exact binomial against p = 0.5, over decided windows):

    arm wins,  p < 0.05   -> the arm is better
    baseline wins, p < 0.05 -> the baseline is better
    p >= 0.05             -> INCONCLUSIVE
    ties > 60% of windows -> INDISTINGUISHABLE

A unanimous result is exactly what a file-pairing error produces, so run the
integrity check (`check_integrity.py`) before reading anything into it.

Usage:
  python score.py --arm b05j0
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from math import comb
from pathlib import Path


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0
    pk = [comb(n, i) * p**i * (1 - p)**(n - i) for i in range(n + 1)]
    return min(1.0, sum(v for v in pk if v <= pk[k] * (1 + 1e-12)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--baseline", default="m17")
    ap.add_argument("--windows", default=None)
    ap.add_argument("--record", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    doc = json.loads(Path(a.windows or f"windows/{a.arm}.json").read_text())
    index = {w["id"]: w for w in doc["windows"]}
    text = Path(a.record or f"judgement/{a.arm}/RECORD.md").read_text()

    votes = {}
    for line in text.splitlines():
        m = re.match(r"\|\s*([A-Z]\d+)\s*\|\s*(\w*)\s*\|", line.strip())
        if m and m.group(2):
            votes[m.group(1)] = m.group(2).strip().lower()

    missing = [i for i in index if i not in votes]
    if missing:
        print(f"ABORT: {len(missing)} windows with no verdict: {missing[:8]}")
        print("  the criterion needs all of them recorded before unblinding.")
        return 1
    bad = {i: v for i, v in votes.items() if v not in ("a", "b", "tie", "empate")}
    if bad:
        print(f"ABORT: invalid verdicts: {bad}")
        return 1

    result = []
    for i, v in votes.items():
        w = index[i]
        if v in ("tie", "empate"):
            winner = "tie"
        else:
            winner = w["A"] if v == "a" else w["B"]
        result.append((i, winner))

    tot = Counter(w for _, w in result)
    ties = tot["tie"]
    frac_ties = ties / len(result)
    n = tot[a.arm] + tot[a.baseline]
    p = binom_two_sided(tot[a.arm], n)

    print("=" * 62)
    print(f"{a.arm} against {a.baseline} -- sealed criterion")
    print("=" * 62)
    print(f"\n{len(result)} windows: {a.arm} {tot[a.arm]}, "
          f"{a.baseline} {tot[a.baseline]}, tie {ties} ({frac_ties*100:.0f}%)")
    print(f"decided: {n}   p = {p:.4f}")

    print("\n" + "=" * 62)
    if frac_ties > 0.60:
        verdict = "INDISTINGUISHABLE"
        print(f"VERDICT: INDISTINGUISHABLE ({frac_ties*100:.0f}% ties)")
    elif p >= 0.05:
        verdict = "INCONCLUSIVE"
        print(f"VERDICT: INCONCLUSIVE (p = {p:.4f})")
    elif tot[a.arm] > tot[a.baseline]:
        verdict = f"{a.arm} better"
        print(f"VERDICT: {a.arm} wins, p = {p:.4f}")
    else:
        verdict = f"{a.baseline} better"
        print(f"VERDICT: {a.baseline} wins, p = {p:.4f}")
    print("=" * 62)

    print("\nOne judge, one seed, windows reused across experiments.")
    print("See the Limits section of the README before reading further.")

    out = Path(a.out or f"results/{a.arm}_score.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "arm": a.arm, "baseline": a.baseline, "verdict": verdict,
        "counts": {a.arm: tot[a.arm], a.baseline: tot[a.baseline], "tie": ties},
        "decided": n, "p": p,
        "per_window": [{"id": i, "winner": w} for i, w in result]}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
