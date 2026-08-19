# Harness

Four scripts. Every arm goes through the same four steps, and the guards are
the point: the experiment is only worth anything if exactly one thing changed
and the two sides really are different models.

```
prepare_arm.py     generate the config, verify by diff that one variable moved,
                   draw the A/B assignment for the windows
                        ↓  seal with sha256, then train
make_pairs.py      run inference for both arms, cut the blind crops, write the
                   judging sheet
                        ↓  fill in the sheet, one verdict per window
check_integrity.py confirm the two arms are distinct models   ← before reading
score.py           unblind and apply the sealed criterion
```

## The order matters

`prepare_arm.py` **aborts** if the `NetworkFromConfig` dump changes. That is not
paranoia: at 256 px in XY the `autoconfigure` adds a seventh pooling stage, so
"bigger patch" would silently become "bigger patch and deeper network". It also
aborts if any config field beyond the expected set moved.

Seal the config, the windows and the preregistration with `sha256sum` **before**
training. After the numbers exist, the criterion cannot be adjusted — that is
the whole point of fixing it first.

`make_pairs.py` **aborts** if the two arms sit at different training steps, and
never selects `best_val_balanced_accuracy`.

`check_integrity.py` runs **before** `score.py`, not after. A unanimous verdict
is exactly what a wrong symlink produces.

## Judging

The sheet written by `make_pairs.py` states the criterion, including what does
**not** count as a win. That was learned the hard way: an earlier experiment in
this series left it implicit, the judge counted small smoothness differences as
wins, and the verdict had to be qualified afterwards. The criterion belongs in
the preregistration, not in the conversation.

Do not open the window file before the sheet is complete. It holds the A/B map.

## Reproducing an arm from scratch

```bash
# 1. prepare and seal
python prepare_arm.py --arm b09 --slices 9 --seed-ab 202608179
sha256sum configs/b09.json windows/b09.json prereg/B9.md

# 2. train (~5 h on one RTX 5070 at batch 64)
python -m koine_machines.training.train configs/b09.json

# 3. inference and blind crops
python make_pairs.py --arm b09

# 4. judge: fill in judgement/b09/RECORD.md

# 5. verify, then score
python check_integrity.py --arm b09
python score.py --arm b09
```

The baseline (`m17`) is trained once and reused by every arm, which is why each
new point costs one run rather than two. `make_pairs.py --reuse-baseline` will
symlink its predictions instead of running inference again.

## Inputs

The arms read the pooled 9.6 µm isotropic inputs produced by the official
`prepare_9um_isotropic_input.py` recipe from the public 2.399 µm surface
volumes, and the labels from the `scrollprize` Hugging Face bucket. Nothing here
needs credentials.
