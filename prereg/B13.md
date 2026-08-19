# PREREG B13 — depth window in z: 13 slices against 17

**Date:** 2026-08-16
**Sealed:** `27432e4dc78086d8feabebf9fcd9857ec0b5913e343f55545df9455f11a72a00`
**Baseline:** the 17-slice arm (`m17`), already trained. Not retrained.

---

## 1. Origin and hypothesis

`err`, 2026-08-14: *"choosing 17 slices isn't really optimized, it probably
isn't the best number"*, and *"we could be stepping too coarsely, for instance.
even more at 9um"*. An explicit recommendation from the person who wrote the
release, about a number nobody had tested.

    17 slices × 9.596 µm = 163.1 µm
    13 slices × 9.596 µm = 124.7 µm

    H_thin: ink and useful signal occupy less than 163 µm; the outer slices
            bring in neighbouring papyrus and adjacent sheets, which dilute.
            13 > 17 in readability.

    H_wide: depth context helps the model separate ink from background;
            cutting the window loses information. 13 ≤ 17.

## 2. Geometry, verified in the code

`_surface_patch_bbox` builds the bbox with `z0 = surface − depth // 2`, where
`depth = patch_size[0]`. `_flat_z_window_bbox` requires
`base_depth − window_depth` to be even, trims `depth_reduction // 2` from each
side, and **applies the offset to the whole bbox** — it shifts the window
without changing its depth, and is not clamped by the trim.

With a 21-slice pooled input and `surface = 10`:

| arm | patch_size[0] | bbox | offset ±2 | inside [0,21]? |
|---|---|---|---|---|
| m17 | 17 | 2–19 | 0–17 / 4–21 | yes, at the limit |
| b13 | 13 | 4–17 | 2–15 / 6–19 | yes |

`21 = 17 + 2·2`: the release sized the 21-slice pooling so the ±2 jitter fits
exactly. Testing **wider** than 17 would need a new 25-slice pooled input.

**The target does not change:** only the centre slice of the label carries
content, and it lies inside both windows.

## 3. Arms and budget

One new run: **b13**, seed 42, identical to m17 except `patch_size[0]` and
`window_depth`. The `NetworkFromConfig` dump is identical across all 40 lines —
only the stem's `in_channels` differs.

Estimated cost: ~6 h on one RTX 5070 at batch 64.

**Second pass, conditional:** seed 7 on both arms, only if the first pass shows
a difference. If it comes back inconclusive, the experiment ends and the second
seed is not spent. Declared in advance; deciding afterwards would be a post-hoc
choice.

## 4. Windows — 60, single stratum

**N = 60 windows of 256 × 768 px**, drawn with seed **20260817** before
training.

**Selector: surface coverage ≥ 90% in the pooled input, not annotation.** The
coverage is **identical in both arms** by construction — same volume, same
plane range, only the operator differs — so the draw is **blind to the arm**.

**Window shape:** the unit in `sean`'s criterion is the line of text ("each
letter within a row"); a square window cuts the line. 768 px at 9.6 µm ≈ 7.4 mm.

**Cap of 4 windows per physical segment.** Single stratum: every window needs
≥ 3% `inklabels`, so there is something to read.

## 5. Verdict criterion

Two-sided binomial against p = 0.5 over the decided windows.

| result | verdict |
|---|---|
| b13 wins, p < 0.05 | H_thin supported. Triggers the second seed. |
| m17 wins, p < 0.05 | H_wide supported. Triggers the second seed. |
| p ≥ 0.05 | INCONCLUSIVE. Ends. |
| ties > 60% | INDISTINGUISHABLE. Ends. |

**The first pass never closes a positive verdict on its own.**

## 6. What is not the criterion

`val_loss` (banned: an earlier experiment in this project measured that it
anti-correlates with what the eye sees), `val_balanced_accuracy` (the official
`best_checkpoint_metric`; recorded, does not decide), `val_bce_unsmoothed`
(recorded, does not decide).

## 7. Declared confounders

**7.1 — A narrower window sees less data per patch.** That is the mechanism of
H_wide, not a flaw in the test. Stated so a defeat cannot be reinterpreted
afterwards.

**7.2 — The baseline is reused.** m17 was trained with the same recipe, seed and
corpus, but not specifically for this experiment. Both checkpoints at the same
step.

**7.3 — Single stratum.** Nothing is seen outside the annotated regions.

**7.4 — One point pair.** 13 against 17 does not say where the optimum is.

**7.5 — Region-held-out validation.** Nothing here speaks to generalisation
across segments or scrolls.
