"""Fed-ISIC2019: images that appear twice under different ISIC IDs
(2026-10-16; decision 2026-10-16: keep FLamby's split
as published and disclose).

Duplicates are found as identical cached features
(extract_isic_features.py: every image gets the same deterministic transform,
so identical files give identical features) and then confirmed by SHA-256 of
the original and the preprocessed image files.

Writes (results_tables/):
  isic_duplicates.csv     one row per duplicate pair: centre, split, ISIC ID
                          and label of both images, whether the files are
                          byte-identical
  isic_leaked_test.csv    the test images with a copy in a training set
                          (the list the clean-run sensitivity check excludes)
Prints the leaked test images by centre and class, and how many duplicate
pairs inside a training set straddle the learning-rate screen's
train/validation split (seeds 0-19, 10% per centre).

Usage: python isic_duplicates.py   (about 10 s; reads no test labels beyond
       those already in the cached feature file)
"""
import hashlib
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bench.data import train_val_split  # noqa: E402
from bench.tasks import isic_features  # noqa: E402

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
DATA = Path("data/flamby/fed_isic2019")
OUT = Path("results_tables")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original(name):
    """The original ISIC 2019 file (some IDs carry a _downsampled suffix)."""
    (p,) = (DATA / "ISIC_2019_Training_Input").glob(f"{name}*.jpg")
    return p


def images():
    """One row per image: split, centre, row index within the centre, ID, label."""
    d = isic_features()
    rows, feats = [], []
    for split in ("train", "test"):
        for c, (X, y, names) in enumerate(d[split]):
            for i, (name, label) in enumerate(zip(names, y.tolist())):
                rows.append(dict(split=split, centre=c, row=i, name=name, label=label))
            feats.append(X.numpy())
    return pd.DataFrame(rows), np.concatenate(feats)


def pairs(df, X):
    groups = defaultdict(list)
    for k, x in enumerate(X):
        groups[x.tobytes()].append(k)
    groups = [g for g in groups.values() if len(g) > 1]
    assert all(len(g) == 2 for g in groups), "a duplicate group with more than 2 images"
    out = []
    for a, b in groups:
        a, b = sorted((df.iloc[a], df.iloc[b]), key=lambda r: (r.split != "train", r["name"]))
        same = [sha(f(r["name"])) for r in (a, b)
                for f in (original, lambda n: DATA / "ISIC_2019_Training_Input_preprocessed" / f"{n}.jpg")]
        out.append(dict(centre_a=a.centre, split_a=a.split, name_a=a["name"], label_a=CLASSES[a.label],
                        row_a=a.row, centre_b=b.centre, split_b=b.split, name_b=b["name"],
                        label_b=CLASSES[b.label], row_b=b.row,
                        same_label=a.label == b.label,
                        original_identical=same[0] == same[2],
                        preprocessed_identical=same[1] == same[3]))
    return pd.DataFrame(out).sort_values(["split_a", "split_b", "centre_a", "name_a"])


def main():
    df, X = images()
    p = pairs(df, X)
    assert p.original_identical.all() and p.preprocessed_identical.all()
    kind = p.split_a + "/" + p.split_b
    print(f"{len(p)} duplicate pairs, all byte-identical: {kind.value_counts().to_dict()}; "
          f"different centres: {(p.centre_a != p.centre_b).sum()}; "
          f"different labels: {(~p.same_label).sum()}")

    leak = p[kind == "train/test"]
    leak = pd.DataFrame(dict(centre=leak.centre_b, test_name=leak.name_b, test_label=leak.label_b,
                             train_name=leak.name_a, train_label=leak.label_a,
                             same_label=leak.same_label)).sort_values(["centre", "test_label"])
    OUT.mkdir(exist_ok=True)
    p.to_csv(OUT / "isic_duplicates.csv", index=False)
    leak.to_csv(OUT / "isic_leaked_test.csv", index=False)
    print(f"\n{len(leak)} test images with a copy in training (of {(df.split == 'test').sum()}):")
    t = pd.crosstab(leak.centre, pd.Categorical(leak.test_label, CLASSES), dropna=False,
                    margins=True, margins_name="total")
    print(t.to_string())
    print(leak[~leak.same_label].to_string(index=False))

    # Within-training duplicates that straddle the screen's train/validation split
    tt = p[(kind == "train/train")]
    assert (tt.centre_a == tt.centre_b).all()
    d = isic_features()
    split = []
    for seed in range(20):
        n = 0
        for c, (_, y, _) in enumerate(d["train"]):
            va = set(train_val_split(y.numpy(), 0.1, seed + c)[1].tolist())
            g = tt[tt.centre_a == c]
            n += sum((ra in va) != (rb in va) for ra, rb in zip(g.row_a, g.row_b))
        split.append(n)
    print(f"\nscreen: training pairs split across train/validation, seeds 0-19: {split} "
          f"(validation: 1,859 images)")
    print(f"saved {OUT / 'isic_duplicates.csv'}, {OUT / 'isic_leaked_test.csv'}")


if __name__ == "__main__":
    main()
