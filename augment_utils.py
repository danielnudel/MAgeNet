"""Shared augmentation logic for single-marker and concatenated multi-marker CSVs."""
import json
import os

import numpy as np
import pandas as pd

READS_PER_AUGMENTED_SAMPLE = 8192
NUM_SUB_SAMPLES = 128


def augment(data_path, include_age=True):
    """Augment a (single- or multi-marker) processed CSV.

    For multi-marker CSVs a manifest JSON must exist alongside the file
    (same stem, _manifest.json suffix).  Each marker section is augmented
    independently via multinomial resampling; the results are combined into
    the same concatenated column layout.

    include_age — set False for predict.py which does not need the age column.
    """
    data = pd.read_csv(data_path, dtype={"tag": str})
    manifest_path = os.path.splitext(data_path)[0] + "_manifest.json"

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        return _augment_concat(data, manifest, include_age)
    else:
        return _augment_single(data, include_age)


# ──────────────────────────────────────────────────────────────────────────────
# Single-marker
# ──────────────────────────────────────────────────────────────────────────────

def _augment_single(data, include_age):
    ages, total_reads_list, tags = [], [], []
    boot_rows = np.zeros((len(data) * NUM_SUB_SAMPLES, len(data.columns) - 3))
    row_counter = 0

    for _, row in data.iterrows():
        age = row["age"]
        total_reads = row["total_reads_origin"]
        tag = row["tag"]
        probabilities = row.values[:-3].astype(float) / total_reads
        for _ in range(NUM_SUB_SAMPLES):
            boot_rows[row_counter] = np.random.multinomial(
                READS_PER_AUGMENTED_SAMPLE, probabilities
            )
            ages.append(age)
            total_reads_list.append(total_reads)
            tags.append(tag)
            row_counter += 1

    df = pd.DataFrame(boot_rows, columns=data.columns[:-3])
    fixed_columns = df.columns
    df["age"] = ages
    df["total_reads_origin"] = total_reads_list
    df["tag"] = tags

    num_sites = len(data.columns[0])
    for i in range(num_sites + 1):
        cols = [c for c in fixed_columns if c.count("C") == i]
        df[f"C_count_{i}"] = df[cols].sum(axis=1)
    for site in range(num_sites):
        cols = [c for c in fixed_columns if c[site] == "C"]
        df[f"site_{site + 1}"] = df[cols].sum(axis=1)

    col_order = (
        [c for c in df.columns if "site_" in c]
        + [c for c in df.columns if "C_count_" in c]
        + sorted(data.columns[:-3].tolist())
    )
    tail = ["tag", "age"] if include_age else ["tag"]
    return df[col_order + tail]


# ──────────────────────────────────────────────────────────────────────────────
# Multi-marker (concatenated)
# ──────────────────────────────────────────────────────────────────────────────

def _augment_concat(data, manifest, include_age):
    n_rows = len(data) * NUM_SUB_SAMPLES
    tags, ages = [], []
    for _, row in data.iterrows():
        for _ in range(NUM_SUB_SAMPLES):
            tags.append(row["tag"])
            ages.append(row["age"])

    marker_dfs = []

    for entry in manifest:
        marker = entry["marker"]
        num_sites = entry["num_sites"]
        total_reads_col = entry["total_reads_col"]
        pat_cols = sorted(c for c in data.columns if c.startswith(f"{marker}__"))

        boot_rows = np.zeros((n_rows, len(pat_cols)))
        row_counter = 0
        for _, row in data.iterrows():
            probs = row[pat_cols].values.astype(float) / row[total_reads_col]
            for _ in range(NUM_SUB_SAMPLES):
                boot_rows[row_counter] = np.random.multinomial(
                    READS_PER_AUGMENTED_SAMPLE, probs
                )
                row_counter += 1

        df_m = pd.DataFrame(boot_rows, columns=pat_cols)

        # Per-marker derived features, also prefixed
        for i in range(num_sites + 1):
            cols = [c for c in pat_cols if c.split("__", 1)[1].count("C") == i]
            df_m[f"{marker}__C_count_{i}"] = df_m[cols].sum(axis=1)
        for site in range(num_sites):
            cols = [c for c in pat_cols if c.split("__", 1)[1][site] == "C"]
            df_m[f"{marker}__site_{site + 1}"] = df_m[cols].sum(axis=1)

        col_order = (
            [c for c in df_m.columns if "__site_" in c]
            + [c for c in df_m.columns if "__C_count_" in c]
            + sorted(pat_cols)
        )
        marker_dfs.append(df_m[col_order])

    result = pd.concat(marker_dfs, axis=1)
    result["tag"] = tags
    if include_age:
        result["age"] = ages
    return result
