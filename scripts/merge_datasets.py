#!/usr/bin/env python3
"""Merge multiple JSONL datasets, removing duplicates."""

import json
import sys
from pathlib import Path


def load_jsonl(path: str) -> list:
    """Load JSONL file and return list of samples."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def merge_datasets(input_paths: list, output_path: str):
    """Merge multiple datasets, deduplicating by ID."""
    all_samples = []
    seen_ids = set()

    for path in input_paths:
        print(f"Loading {path}...")
        samples = load_jsonl(path)
        print(f"  Found {len(samples)} samples")

        for sample in samples:
            sample_id = sample.get("id", "")
            if sample_id and sample_id not in seen_ids:
                all_samples.append(sample)
                seen_ids.add(sample_id)

    # Sort by ID for consistent output
    all_samples.sort(key=lambda x: x.get("id", ""))

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nMerged {len(all_samples)} unique samples to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_datasets.py <output.jsonl> <input1.jsonl> <input2.jsonl> ...")
        sys.exit(1)

    output_path = sys.argv[1]
    input_paths = sys.argv[2:]

    merge_datasets(input_paths, output_path)
