#!/usr/bin/env python3
"""Convert published methylation txt files to CSV format expected by train.py / predict.py.

Input txt format (tab-separated, no header):
  age<float>_<sample_tag>  <sequence>  <count>

Output CSV format:
  one column per unique sequence pattern, total_reads_origin, age, tag (integer)
"""
import argparse
import os
import sys
from collections import defaultdict

import pandas as pd


def parse_sample_id(sample_id):
    """Parse 'age<float>_<tag>' -> (age_float, tag_str)."""
    if not sample_id.startswith("age"):
        raise ValueError(f"Unexpected sample ID format: {sample_id!r}")
    rest = sample_id[3:]
    sep = rest.index("_")
    return float(rest[:sep]), rest[sep + 1:]


def extract_subseq(sequence, positions):
    """Return characters at the given 1-indexed positions."""
    return "".join(sequence[p - 1] for p in positions)


def convert(input_path, positions, output_path):
    # samples preserves insertion order (Python 3.7+)
    samples = {}

    with open(input_path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                print(f"Line {lineno}: expected 3 tab-separated columns, got {len(parts)} — skipping",
                      file=sys.stderr)
                continue
            sample_id, sequence, count_str = parts
            try:
                count = int(count_str)
            except ValueError:
                print(f"Line {lineno}: non-integer count {count_str!r} — skipping", file=sys.stderr)
                continue

            try:
                age, tag_str = parse_sample_id(sample_id)
            except ValueError as exc:
                print(f"Line {lineno}: {exc} — skipping", file=sys.stderr)
                continue

            if positions:
                if max(positions) > len(sequence):
                    print(f"Line {lineno}: position {max(positions)} exceeds sequence length "
                          f"{len(sequence)} — skipping", file=sys.stderr)
                    continue
                subseq = extract_subseq(sequence, positions)
            else:
                subseq = sequence

            if not all(c in "CT" for c in subseq):
                continue

            if tag_str not in samples:
                samples[tag_str] = {"age": age, "counts": defaultdict(int), "total": 0}
            samples[tag_str]["counts"][subseq] += count
            samples[tag_str]["total"] += count

    if not samples:
        print("No samples parsed — output not written.", file=sys.stderr)
        sys.exit(1)

    all_patterns = sorted({pat for s in samples.values() for pat in s["counts"]})

    rows = []
    for tag_str, s in samples.items():
        row = {pat: s["counts"].get(pat, 0) for pat in all_patterns}
        row["total_reads_origin"] = s["total"]
        row["age"] = s["age"]
        row["tag"] = tag_str
        rows.append(row)

    columns = all_patterns + ["total_reads_origin", "age", "tag"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} samples × {len(all_patterns)} patterns to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert published methylation txt to CSV for train.py / predict.py"
    )
    parser.add_argument("input", help="Input .txt file")
    parser.add_argument(
        "-p", "--positions",
        help="Comma-separated 1-indexed positions to extract from each sequence (e.g. 2,3)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV path (default: input path with .csv extension)",
    )
    args = parser.parse_args()

    positions = [int(p) for p in args.positions.split(",")] if args.positions else None

    if args.output:
        output_path = args.output
    else:
        basename = os.path.splitext(os.path.basename(args.input))[0]
        chrom, coords = basename.split(":", 1)  # "chr10", "22623318-22623477"
        output_path = os.path.join("data", "processed", chrom, coords + ".csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    convert(args.input, positions, output_path)


if __name__ == "__main__":
    main()
