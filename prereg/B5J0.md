# PREREG B5J0 — what broke B5: lack of window, or excessive jitter?

**Date:** 2026-08-18
**Sealed:** `b317235aa0877269d8accf391f11d77deed019d81a9d9f7a1aacc2dcb1f1bef6`
**Origin:** §6.2 of `B5.md` declared this experiment necessary **if B5 broke**.
It broke.

---

## 1. The problem this resolves

The depth axis closed with a located breaking point:

| window | slices | verdict | judgement |
|---|---|---|---|
| 48.0 µm | 5 | **breaks** | m17 wins 58–0, 2 ties, p = 0.0000 |
| 86.4 µm | 9 | indistinguishable | 60 ties out of 60 |
| 124.7 µm | 13 | indistinguishable | 60 ties out of 60 |
| 163.1 µm | 17 | reference | — |

b05 loses confidence on ink pixels measurably: p99 from 188→68 on
`pherc0139-w039`, 196→79 on `pherc1667-w013`, 199→103 on `pherc0814-46527`,
against only 200→173 and 201→182 on two Paris 4 segments.

**But the B5 design does not separate two possible causes**, and this was
declared in its §6.2 before measuring:

    C_window: 48.0 µm is too little depth. The model loses context it needs to
              tell ink from background.

    C_jitter: `flat_z_window_jitter` with max_offset 2 shifts the window by up
              to 40% of its own width when the window has 5 slices (against 12%
              at 17). The augmentation stops being a perturbation and starts
              changing the content; the model trains against a moving target.

Both predict exactly what was observed. **Without separating them, the claim
"48 µm is too little window" may be wrong** — the true one being "±2 jitter is
too large for a 5-slice window".

## 2. The experiment

One run: **b05j0** — identical to b05 except
`flat_z_window_jitter.max_offset`, 2 → **0**.

With `max_offset: 0` the jitter is off (`_flat_z_window_bbox` returns the
canonical bbox with no shift, verified in the code). The window still has 5
slices, centred, unperturbed.

**Compared against m17**, the same baseline as the other three points.

| result | reading |
|---|---|
| b05j0 **also breaks** | **C_window.** 48.0 µm is too little depth regardless of the jitter. The axis breaking point is real. |
| b05j0 **does not break** | **C_jitter.** The 48 µm window is enough; what broke b05 was disproportionate jitter. The axis goes further than B5 suggested, and the official recipe has an undocumented coupling between depth and augmentation. |
| intermediate | both causes contribute. Declared as grey zone (§5). |

**No outcome is uninformative, and the second is more interesting than the
first** — it would be a finding about the recipe, not about the parameter.

## 3. Central confounder, declared

**Turning the jitter off removes augmentation.** b05j0 trains with less
regularisation than m17, which disadvantages it by a route that is not the
window. If b05j0 breaks, part of that may be lost augmentation rather than lost
depth.

**This is unavoidable in this design** and limits the conclusion: `C_window` is
only established if b05j0 breaks **at the same magnitude** as b05. If it breaks
less, the reading is mixed.

**The control that would resolve it, recorded as the next step if needed:** an
**m17j0** arm — 17 slices with `max_offset: 0` — would measure the cost of
turning the jitter off in isolation, with sufficient window. ~6 h, justified
only if b05j0 comes back intermediate.

## 4. Windows — the same 60, fifth pass

New A/B assignment, seed 202608180.

**Declared risk, now high:** the judge has assessed these windows four times. On
the fifth, memory of the content is near certain — and, worse, he already knows
that in one round the narrow arm visibly lost. That can bias expectation.

**Mitigation:** the A/B assignment is redrawn and he never knew which side was
which in any round. But the expectation bias is real and does not go away.
**Accepted and declared limit.** If the result lands near the threshold, this
alone is reason not to conclude.

## 5. Verdict criterion

Two-sided binomial against p = 0.5 over the decided windows.

| result | verdict |
|---|---|
| m17 wins, p < 0.05, **with ≥ 40 decided** | **C_window supported.** The breaking point belongs to depth. |
| p ≥ 0.05, or ties > 60% | **C_jitter supported.** 48 µm is enough when the window is stable. |
| m17 wins, p < 0.05, **with < 40 decided** | **GREY ZONE** — it broke, but less than b05 (which had 58 decided). Both causes contribute. |

The cut at 40 decided is fixed before measuring, and exists to distinguish
"broke the same" from "broke less".

## 6. Label version — a necessary decision

On 2026-08-18 at 11:01 `err` announced that the `ink_9um` masks and labels had
been uploaded **with the alignment pipeline's seam regions masked off by
mistake**, and that he was replacing the files in place.

Verified against our 2026-08-15 copy, on a 1000×1000 crop centred on each mask's
centroid: the remote **only adds** pixels — 45,350 on `pherc0139-w016`, 100 on
`pherc1667-w029`, none on `pherc0814-46527`. Unequal across segments.

**All four depth-axis arms trained on the earlier version.** The download of the
corrected labels was in progress and incomplete when this was written.

**Decision, declared before sealing:** b05j0 points at the pre-correction label
directory — **the same version m17 read**. Without this, b05j0 would train on
different supervision from the baseline and the experiment would have two
variables instead of one.

Consequence: the config differs from b05 in **four** fields — `description`,
`flat_z_window_jitter`, `out_dir` and `datasets` (the `segments_path`). The
`datasets` difference **is not an experimental variable**: it is the same label
content b05 and m17 used, with the path made explicit rather than implicit.

**Limitation this imposes on the whole axis:** all five arms use labels with a
known defect, corrected by the team on 2026-08-18. The internal comparisons
remain valid because the defect is common to all of them. Comparability with
models trained after the correction is not.

## 7. Scope limit

This experiment does **not** revise the B5 verdict, which is scored and sealed.
It answers a different question: which of the two causes produced that verdict.
