#!/usr/bin/env python3
"""Concatenate per-marker CSVs into a single combined CSV for multi-marker training.

Each marker's pattern columns are prefixed with the marker name to avoid collisions
(e.g. ELOVL2_6__TTTTTTTTT). A manifest JSON is written alongside the output CSV so
that the augmentation step knows which columns belong to which marker and can
augment them independently.

Usage:
  python3 scripts/concat_markers.py \\
      ELOVL2_6=data/processed/chr6/11044843-11044997_train.csv \\
      C1orf132=data/processed/chr1/xxx_train.csv \\
      -o data/processed/combined/ELOVL2_6_C1orf132_train.csv
"""
import argparse
import json
import os
import sys

import pandas as pd

METADATA_COLS = {"tag", "age", "total_reads_origin"}


def load_csv(path):
    df = pd.read_csv(path, dtype={"tag": str})
    for col in METADATA_COLS:
        if col not in df.columns:
            print(f"CSV {path!r} is missing required column '{col}'", file=sys.stderr)
            sys.exit(1)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate per-marker CSVs into a single combined CSV"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="MARKER=PATH",
        help="Two or more MARKER=path pairs (e.g. ELOVL2_6=train.csv)",
    )
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    if len(args.inputs) < 2:
        print("At least two MARKER=PATH pairs are required", file=sys.stderr)
        sys.exit(1)

    marker_dfs = []
    for token in args.inputs:
        if "=" not in token:
            print(f"Expected MARKER=PATH, got: {token!r}", file=sys.stderr)
            sys.exit(1)
        marker, path = token.split("=", 1)
        df = load_csv(path)
        marker_dfs.append((marker, df))

    # Start with tag + age from the first marker; inner-join each subsequent marker
    combined = marker_dfs[0][1][["tag", "age"]].copy()
    manifest = []

    for marker, df in marker_dfs:
        pattern_cols = [c for c in df.columns if c not in METADATA_COLS]
        if not pattern_cols:
            print(f"Marker {marker}: no pattern columns found", file=sys.stderr)
            sys.exit(1)

        num_sites = len(pattern_cols[0])  # all patterns have the same length

        # Prefix pattern columns to avoid cross-marker name collisions
        df = df.rename(columns={c: f"{marker}__{c}" for c in pattern_cols})
        prefixed_cols = [f"{marker}__{c}" for c in pattern_cols]

        # Give total_reads_origin a marker-specific name
        total_reads_col = f"total_reads_origin_{marker}"
        df = df.rename(columns={"total_reads_origin": total_reads_col})

        join_cols = prefixed_cols + [total_reads_col]
        combined = combined.merge(df[["tag"] + join_cols], on="tag", how="inner")

        manifest.append({
            "marker": marker,
            "num_sites": num_sites,
            "total_reads_col": total_reads_col,
        })

    n_dropped = len(marker_dfs[0][1]) - len(combined)
    if n_dropped:
        print(f"Note: {n_dropped} sample(s) dropped (not present in all markers)")

    # Column order: for each marker — sorted pattern cols + total_reads — then age, tag
    col_order = []
    for m in manifest:
        pat = sorted(c for c in combined.columns if c.startswith(f"{m['marker']}__"))
        col_order += pat + [m["total_reads_col"]]
    col_order += ["age", "tag"]
    combined = combined[col_order]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    combined.to_csv(args.output, index=False)

    manifest_path = os.path.splitext(args.output)[0] + "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Written {len(combined)} samples × {len(combined.columns)} columns → {args.output}")
    print(f"Manifest → {manifest_path}")
    for m in manifest:
        n = sum(1 for c in combined.columns if c.startswith(f"{m['marker']}__"))
        print(f"  {m['marker']}: {n} pattern columns, {m['num_sites']} sites")


if __name__ == "__main__":
    main()
