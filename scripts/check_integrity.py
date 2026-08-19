#!/usr/bin/env python3
"""
check_integrity.py -- confirm the two arms really are different models.

Run this BEFORE reading any verdict. A unanimous result is exactly what a
file-pairing error produces: same checkpoint under two names, a symlink pointing
at the wrong file, identical inputs. This project has found five of its own
artifacts that passed every internal check and were wrong where it mattered, so
the check is mandatory rather than optional.

What it prints, per segment:
  r          correlation between the two full prediction maps
  |d|        mean absolute difference in levels
  p99        how confident each arm is on its brightest pixels -- this is the
             number that quantifies "the letters look washed out"
  identical  must be False

Usage:
  python check_integrity.py --arm b05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--baseline", default="m17")
    ap.add_argument("--windows", default=None)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    doc = json.loads(Path(a.windows or f"windows/{a.arm}.json").read_text())
    preds = Path(a.preds or f"preds/{a.arm}")

    seen, rows = set(), []
    print(f"{'segment':<20} {'r':>8} {'|d|':>7} {'p99 base':>9} "
          f"{'p99 arm':>8} {'identical':>10}")
    for w in doc["windows"]:
        s = w["segment"]
        if s in seen or len(seen) >= a.n:
            continue
        seen.add(s)
        pa = preds / f"{s}_{a.baseline}.tif"
        pb = preds / f"{s}_{a.arm}.tif"
        if not pa.exists() or not pb.exists():
            print(f"{s:<20} missing prediction")
            continue
        A = tifffile.imread(pa).astype(np.float32)
        B = tifffile.imread(pb).astype(np.float32)
        same = np.array_equal(A, B)
        r = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
        rows.append({"segment": s, "r": r, "d": float(np.abs(A - B).mean()),
                     "p99_base": float(np.percentile(A, 99)),
                     "p99_arm": float(np.percentile(B, 99)), "identical": same})
        print(f"{s:<20} {r:>8.4f} {np.abs(A-B).mean():>7.2f} "
              f"{np.percentile(A,99):>9.1f} {np.percentile(B,99):>8.1f} "
              f"{str(same):>10}")

    print()
    if any(r["identical"] for r in rows):
        print("FAIL: at least one pair is byte-identical. The arms are not")
        print("distinct models -- check the checkpoints and the symlinks before")
        print("reading any verdict.")
        return 1
    if rows:
        rr = np.array([r["r"] for r in rows])
        print(f"OK: no identical pairs. r from {rr.min():.4f} to {rr.max():.4f}.")
        pb = np.array([r["p99_base"] for r in rows])
        pa_ = np.array([r["p99_arm"] for r in rows])
        drop = (pa_ - pb) / pb * 100
        print(f"p99 change against the baseline: median {np.median(drop):+.1f}%, "
              f"min {drop.min():+.1f}%, max {drop.max():+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
