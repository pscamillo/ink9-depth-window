# Preregistrations

Each file was sealed with `sha256sum` **before** the corresponding model was
trained, together with its config and its window file. After the numbers
existed, no criterion was changed.

| file | arm | sha256 (first 8) | verdict |
|---|---|---|---|
| `B13.md` | 13 slices, jitter ±2 | `27432e4d` | indistinguishable, 60/60 ties |
| `B9.md` | 9 slices, jitter ±2 | `cb33793b` | indistinguishable, 60/60 ties |
| `B5.md` | 5 slices, jitter ±2 | `c94c3837` | breaks, 58–0, p = 0.0000 |
| `B5J0.md` | 5 slices, jitter 0 | `b317235a` | indistinguishable, 60/60 ties |
| `L1.md` | corrected labels, 17 slices | `32347bcb` | indistinguishable, 60/60 ties |

## On the judging criterion

From `B5.md` on, the preregistrations state explicitly what does **not** count
as a win: a difference in contrast, smoothness or rounding without a difference
in letter shape or integrity; no identifiable letter on either side.

That was learned the hard way. An earlier experiment in this series left it
implicit, and the judge counted small smoothness differences as wins. The
criterion belongs in the preregistration, not in the conversation.

`B3_FAILED.md` is the preregistration for a 3-slice arm that was sealed and
whose training crashed on the augmentation pipeline. It is kept because the
failure established the implementation floor: `GaussianBlurTransform` with
`blur_sigma=(0.3, 1.5)` and `truncate=6` produces padding up to 4, and
`mode="reflect"` requires padding strictly smaller than the dimension.
