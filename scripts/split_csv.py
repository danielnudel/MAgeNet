#!/usr/bin/env python3
"""Split a processed CSV into train / validation / test sets, stratified by age.

Split ratios
  - 70 % train+val  /  30 % test
  - 70 % train      /  30 % validation  (within train+val)

data/train_test_split.csv is the canonical split registry.  Only tags already
listed there are processed; tags absent from the registry are silently ignored.
To add new tags, first register them in train_test_split.csv (run with --register).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_REGISTRY = os.path.join("data", "train_test_split.csv")
N_AGE_BINS = 10


def load_registry():
    if os.path.exists(SPLIT_REGISTRY):
        return pd.read_csv(SPLIT_REGISTRY, dtype={"tag": str})
    return pd.DataFrame(columns=["tag", "split"])


def save_registry(df):
    os.makedirs(os.path.dirname(SPLIT_REGISTRY), exist_ok=True)
    df.to_csv(SPLIT_REGISTRY, index=False)


def age_strata(ages, n_bins=N_AGE_BINS):
    """Quantile-bin ages into at most n_bins groups; merge rare bins so every
    stratum has at least 2 samples (required by train_test_split)."""
    labels, _ = pd.qcut(ages, q=n_bins, labels=False,
                        retbins=True, duplicates="drop")
    labels = labels.astype(float)

    # Merge any bin with fewer than 2 members into its neighbour
    counts = pd.Series(labels).value_counts()
    rare = counts[counts < 2].index
    for b in sorted(rare):
        neighbour = b + 1 if b + 1 in counts.index else b - 1
        labels[labels == b] = neighbour

    return labels.astype(str)


def assign_splits(df):
    """Return a Series mapping df.index -> 'train'/'validation'/'test'."""
    assignments = pd.Series(index=df.index, dtype=str)

    if len(df) < 5:
        # Too small to stratify — distribute round-robin
        cycle = ["train", "train", "train", "validation", "test"]
        for i, idx in enumerate(df.index):
            assignments[idx] = cycle[i % len(cycle)]
        return assignments

    strata = age_strata(df["age"].values)

    idx_trainval, idx_test = train_test_split(
        df.index, test_size=0.3, stratify=strata, random_state=42
    )
    assignments[idx_test] = "test"

    df_tv = df.loc[idx_trainval]
    strata_tv = age_strata(df_tv["age"].values)

    idx_train, idx_val = train_test_split(
        df_tv.index, test_size=0.3, stratify=strata_tv, random_state=42
    )
    assignments[idx_train] = "train"
    assignments[idx_val] = "validation"

    return assignments


def main():
    parser = argparse.ArgumentParser(
        description="Split processed methylation CSV into train/validation/test"
    )
    parser.add_argument("input", help="Input CSV file (output of txt_to_csv.py)")
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory for split output files (default: same directory as input)",
    )
    parser.add_argument(
        "--register", action="store_true",
        help="Assign splits to tags not yet in the registry and save them",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype={"tag": str})
    for col in ("tag", "age"):
        if col not in df.columns:
            print(f"Input CSV is missing required column '{col}'", file=sys.stderr)
            sys.exit(1)

    registry = load_registry()
    registered_tags = set(registry["tag"])

    if args.register:
        new_mask = ~df["tag"].isin(registered_tags)
        df_new = df[new_mask].reset_index(drop=True)
        print(f"{(~new_mask).sum()} tag(s) already registered — skipped")
        print(f"{new_mask.sum()} new tag(s) to register")
        if not df_new.empty:
            splits = assign_splits(df_new)
            new_entries = pd.DataFrame({"tag": df_new["tag"], "split": splits.values})
            registry = pd.concat([registry, new_entries], ignore_index=True)
            save_registry(registry)
            counts = new_entries["split"].value_counts().to_dict()
            print(f"Registered: {counts} → {SPLIT_REGISTRY}")
    else:
        known_mask = df["tag"].isin(registered_tags)
        n_ignored = (~known_mask).sum()
        if n_ignored:
            print(f"{n_ignored} tag(s) not in registry — ignored")
        print(f"{known_mask.sum()} tag(s) found in registry")

    # Write output files for tags present in both the input CSV and the registry
    tag_to_split = registry.set_index("tag")["split"].to_dict()
    df["_split"] = df["tag"].map(tag_to_split)
    df = df[df["_split"].notna()]

    if df.empty:
        print("No matching tags found — no output files written.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    stem = os.path.splitext(os.path.basename(args.input))[0]

    for name in ("train", "validation", "test"):
        subset = df[df["_split"] == name].drop(columns=["_split"])
        out_path = os.path.join(output_dir, f"{stem}_{name}.csv")
        subset.to_csv(out_path, index=False)
        print(f"  {name:12s}: {len(subset):4d} rows → {out_path}")


if __name__ == "__main__":
    main()
