#!/usr/bin/env python3
"""
make_pairs.py -- run inference for both arms and cut the blind A/B crops.

Two guards that are not optional:

  * both arms must sit at the SAME training step. A comparison between
    different steps measures the step, not the variable.
  * `best_val_balanced_accuracy` is never used. It is not the criterion, and on
    the official release that checkpoint lands around step 15,000 of 78,125.

Note on which file is the last one: `num_iterations: 78125` with
`save_every: 2500` writes no checkpoint at the final step, because the trainer
only saves when `(step+1) % save_every == 0` and the closing call does not force
it. The last file is ckpt_077500.pth, 99.2% of training. The same applies to the
official release with save_every 5000.

Display is (p-64)/128: with bce_label_smoothing 0.5 the confident background
sits near 64, not 0, and without the rescale the maps look like uniform grey
even when they carry signal.

Usage:
  python make_pairs.py --arm b09
  python make_pairs.py --arm b05j0 --reuse-baseline preds/b05
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

INFER = ["--overlap", "0.50", "--blend-mode", "hann",
         "--batch-size", "4", "--num-workers", "4", "--no-compile"]
OFF, SCALE = 64.0, 128.0


def say(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def last_ckpt(d: Path):
    cks = sorted(d.glob("ckpt_*.pth"),
                 key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    return cks[-1] if cks else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--baseline", default="m17")
    ap.add_argument("--windows", default=None)
    ap.add_argument("--inputs", default="renders")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--preds", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reuse-baseline", default=None,
                    help="symlink the baseline .tif from this directory instead "
                         "of running inference again")
    a = ap.parse_args()

    win_p = Path(a.windows or f"windows/{a.arm}.json")
    preds = Path(a.preds or f"preds/{a.arm}")
    out = Path(a.out or f"judgement/{a.arm}")
    doc = json.loads(win_p.read_text())
    windows = doc["windows"]
    segs = sorted({w["segment"] for w in windows})
    say(f"{len(windows)} windows across {len(segs)} segments")

    ckpts, steps = {}, {}
    for arm in (a.baseline, a.arm):
        ck = last_ckpt(Path(a.ckpt_dir) / f"{arm}-seed42")
        if ck is None:
            say(f"ABORT: no ckpt_*.pth for {arm}")
            return 1
        ckpts[arm] = str(ck)
        steps[arm] = int("".join(c for c in ck.stem if c.isdigit()))
        say(f"{arm}: {ck.name} (step {steps[arm]})")
    if steps[a.baseline] != steps[a.arm]:
        say(f"ABORT: different steps. The comparison needs the same step.")
        return 1
    say(f"both at step {steps[a.arm]} -- comparison valid")

    preds.mkdir(parents=True, exist_ok=True)
    if a.reuse_baseline:
        src = Path(a.reuse_baseline)
        n = 0
        for f in src.glob(f"*_{a.baseline}.tif"):
            dst = preds / f.name
            if not dst.exists():
                dst.symlink_to(f.resolve())
                n += 1
        say(f"reused {n} baseline predictions from {src}")

    for arm in (a.baseline, a.arm):
        for s in segs:
            dst = preds / f"{s}_{arm}.tif"
            if dst.exists():
                continue
            say(f"{s} [{arm}]: inferring")
            cmd = [sys.executable, "-m", "koine_machines.inference.infer",
                   f"{a.inputs}/{s}_mean.zarr", ckpts[arm], str(dst)] + INFER
            if subprocess.run(cmd, capture_output=True).returncode != 0:
                say(f"  FAILED on {s} [{arm}]")
                return 1

    import tifffile
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    cache: dict[str, np.ndarray] = {}
    for w in windows:
        s, y, x = w["segment"], w["y"], w["x"]
        h, wd = w["height"], w["width"]
        for side in ("A", "B"):
            k = f"{s}_{w[side]}"
            if k not in cache:
                if len(cache) > 4:
                    cache.pop(next(iter(cache)))
                cache[k] = tifffile.imread(preds / f"{k}.tif")
            crop = cache[k][y:y+h, x:x+wd].astype(np.float32)
            d = (np.clip((crop - OFF) / SCALE, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(d).save(out / f"{w['id']}_{side}.png")
        say(f"{w['id']}: {s} y={y} x={x}")

    sheet = out / "RECORD.md"
    sheet.write_text("\n".join([
        f"# Blind judgement -- {a.arm} against {a.baseline}", "",
        "For each window, open `<id>_A.png` and `<id>_B.png` at full resolution",
        "and record which one **reads better**: `A`, `B` or `tie`.",
        "",
        "Criterion (sean, 2026-07-30): can you see each letter on its own? each",
        "letter within a row? how whole are those letters?",
        "",
        "A difference in contrast, smoothness or rounding WITHOUT a difference",
        "in letter shape or integrity is a TIE.",
        "No identifiable letter on either side is a TIE.",
        "A very straight stroke aligned with its neighbours is suspected FIBRE,",
        "not a letter (err and Blue, 2026-08-16) -- it does not count as",
        "readability.",
        "",
        f"Do NOT open {win_p} before finishing all {len(windows)}.",
        "", "| id | verdict |", "|---|---|",
    ] + [f"| {w['id']} | |" for w in sorted(windows, key=lambda z: z["id"])]) + "\n")

    say(f"\n{len(windows)*2} PNGs in {out}/ and the sheet in {sheet}")
    say(f"Fill it in, then run: python score.py --arm {a.arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
