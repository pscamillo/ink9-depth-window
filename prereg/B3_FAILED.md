# PREREG B3 — 3 slices against 17 — SEALED, TRAINING FAILED

**Date:** 2026-08-18
**Sealed:** `5150c91e83665ecaa7b55ed7c5b4272a6b916f7fc59e03d9ea021bc8cda67187`
**Status:** the run crashed on the augmentation pipeline. Superseded by `B5.md`.

Kept in the repository because the failure established the implementation floor
of this axis, which is a useful result on its own.

---

## What was intended

3 slices (28.8 µm) as the floor of the axis — the assumption being that it was
the minimum still allowing the stem's 3×3×3 convolution without padding in z.
`21 − 3 = 18` is even, so the symmetry guard passes, and the bbox at 9–12 with
±2 jitter stays inside the volume.

## What happened

The run crashed on the first iteration where the Gaussian blur was drawn:

    RuntimeError: Padding size should be less than the corresponding input
    dimension, but got: padding (3, 3) at dimension 1 of input [1, 3, 128, 128]

Stack trace through `augmentation/transforms/noise/gaussian_blur.py`.

## What it established

**The limit is not architectural. It is in the augmentation pipeline.**

The `default` preset includes `GaussianBlurTransform` with
`blur_sigma=(0.3, 1.5)` and `truncate=6`. The kernel size is
`round_to_odd(sigma·6 + 0.5)` and the padding is `ksize // 2`, applied with
`mode="reflect"` — which in PyTorch requires padding **strictly smaller** than
the dimension being padded:

| sigma | ksize | padding | minimum depth |
|---|---|---|---|
| 0.3–0.5 | 3 | 1 | 2 |
| 0.8 | 5 | 2 | 3 |
| 1.0–1.2 | 7 | 3 | 4 |
| **1.4–1.5** | **9** | **4** | **5** |

**With `augmentation_preset: default`, the minimum window depth in z is 5
slices.** At 3, any drawn sigma ≥ 1.0 breaks training.

**The failure mode matters as much as the limit:** the blur is inside a
`OneOfTransform` wrapped in a `RandomTransform`, so it is drawn probabilistically.
The crash does not happen at startup — it happens whenever the draw first lands
on a large enough sigma, which can be minutes or hours into a run.

Anyone testing narrow depth windows on this recipe will hit this. The fix is
either to clamp `ksize` to the available dimension or to fall back from
`reflect` to `replicate` when the padding does not fit.
