#!/usr/bin/env python3
"""
prepare_arm.py -- generate the config and the judging windows for one arm.

Every arm is the 17-slice baseline with exactly one thing changed. This script
generates that config, verifies by diff that nothing else moved, and draws the
A/B assignment for the blind judgement. It trains nothing.

The verification is not decorative: if `autoconfigure` reacts to the new patch
size by adding a pooling stage, the experiment silently becomes two variables.
That happens at 256 px in XY, and it is why the depth axis stops at 5 slices
rather than going wider.

Usage:
  python prepare_arm.py --arm b09 --slices 9  --seed-ab 202608179
  python prepare_arm.py --arm b05 --slices 5  --seed-ab 202608185
  python prepare_arm.py --arm b05j0 --slices 5 --max-offset 0 \
                        --seed-ab 202608180 --labels labels_pre_correction
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

SLICE_UM = 9.596


def network_dump(cfg: dict) -> list[str]:
    from koine_machines.models.make_model import make_model
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        make_model(cfg)
    return [l.strip() for l in buf.getvalue().splitlines() if ":" in l]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="configs/m17.json",
                    help="the 17-slice config every arm derives from")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--slices", type=int, required=True)
    ap.add_argument("--max-offset", type=int, default=None,
                    help="flat_z_window_jitter.max_offset; unchanged if omitted")
    ap.add_argument("--labels", default=None,
                    help="override segments_path (use when the label set changed "
                         "and the baseline must stay comparable)")
    ap.add_argument("--windows-from", default="windows/b13.json",
                    help="reuse the window positions from this file")
    ap.add_argument("--seed-ab", type=int, required=True)
    ap.add_argument("--out-configs", default="configs")
    ap.add_argument("--out-windows", default="windows")
    ap.add_argument("--prefix", default=None, help="window id prefix, e.g. N")
    a = ap.parse_args()

    base = json.loads(Path(a.baseline).read_text())

    # ---- 1. one variable? ----
    cfg = json.loads(json.dumps(base))
    cfg["patch_size"] = [a.slices, 128, 128]
    cfg["flat_z_window_jitter"] = {**base["flat_z_window_jitter"],
                                   "window_depth": a.slices}
    if a.max_offset is not None:
        cfg["flat_z_window_jitter"]["max_offset"] = a.max_offset
    cfg["out_dir"] = f"checkpoints/{a.arm}-seed{base.get('seed', 42)}"
    jit = cfg["flat_z_window_jitter"]["max_offset"]
    cfg["description"] = (
        f"{a.arm}: {a.slices}-slice window ({a.slices*SLICE_UM:.1f} um), "
        f"jitter max_offset {jit}, against the 17-slice baseline. "
        f"Everything else mirrors aligned21_hybrid_3d2d."
    )
    if a.labels:
        for d in cfg["datasets"]:
            d["segments_path"] = str(Path(a.labels).resolve())
        cfg["description"] += (
            f" LABELS: {a.labels} -- the same version the baseline read. "
            f"Needed to keep this to one variable."
        )

    ref, new = network_dump(base), network_dump(cfg)
    print(f"[1] network config: {len(ref)} lines")
    if ref != new:
        print("    ABORT: the network changed. That would be two variables.")
        for x, y in zip(ref, new):
            if x != y:
                print(f"      baseline: {x}\n      {a.arm:>8}: {y}")
        return 1
    print(f"    identical between 17 and {a.slices} slices -- only the stem's "
          f"in_channels differs")

    # ---- 2. what moved ----
    out_cfg = Path(a.out_configs) / f"{a.arm}.json"
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(json.dumps(cfg, indent=1))
    moved = [k for k in base
             if json.dumps(base[k], sort_keys=True) != json.dumps(cfg[k], sort_keys=True)]
    expected = {"description", "patch_size", "flat_z_window_jitter", "out_dir"}
    if a.labels:
        expected.add("datasets")
    print(f"\n[2] wrote {out_cfg}")
    print(f"    fields differing from the baseline: {sorted(moved)}")
    if set(moved) != expected:
        print(f"    ABORT: expected exactly {sorted(expected)}")
        return 1
    print(f"    ok. window {a.slices} slices = {a.slices*SLICE_UM:.1f} um, "
          f"jitter +-{jit} = {jit/a.slices*100:.0f}% of the window")

    # ---- 3. windows ----
    src = json.loads(Path(a.windows_from).read_text())
    rng = np.random.default_rng(a.seed_ab)
    pref = a.prefix or a.arm[0].upper()
    windows = []
    for w in sorted(src["windows"] if "windows" in src else src["janelas"],
                    key=lambda z: z["id"]):
        nw = {k: w[k] for k in ("segmento", "y", "x", "altura", "largura")
              if k in w}
        nw.setdefault("segment", nw.pop("segmento", None))
        nw.setdefault("height", nw.pop("altura", None))
        nw.setdefault("width", nw.pop("largura", None))
        nw["A"] = a.arm if rng.random() < 0.5 else "m17"
        nw["B"] = "m17" if nw["A"] == a.arm else a.arm
        windows.append(nw)
    rng.shuffle(windows)
    for i, w in enumerate(windows):
        w["id"] = f"{pref}{i:02d}"

    out_win = Path(a.out_windows) / f"{a.arm}.json"
    out_win.parent.mkdir(parents=True, exist_ok=True)
    out_win.write_text(json.dumps({
        "arm": a.arm, "seed_ab": a.seed_ab,
        "positions": f"identical to {a.windows_from}",
        "arms": {a.arm: f"{a.slices} slices, {a.slices*SLICE_UM:.1f} um, "
                        f"jitter {jit}",
                 "m17": "17 slices, 163.1 um, jitter 2 (official recipe)"},
        "n": len(windows), "windows": windows}, indent=1))
    n_a = sum(1 for w in windows if w["A"] == a.arm)
    print(f"\n[3] wrote {out_win}: {len(windows)} windows, ids "
          f"{pref}00-{pref}{len(windows)-1:02d}")
    print(f"    A/B (seed {a.seed_ab}): A={a.arm} in {n_a}, "
          f"A=m17 in {len(windows)-n_a}")
    print(f"\nSeal before training:")
    print(f"  sha256sum {out_cfg} {out_win} prereg/{a.arm.upper()}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
