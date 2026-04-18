#!/usr/bin/env python3
"""Analyze the meta-opinions dataset and show statistics."""

import json
import sys
from pathlib import Path
from collections import Counter


def load_dataset(path: str) -> list:
    """Load JSONL dataset."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def analyze_dataset(path: str):
    """Analyze dataset and print statistics."""
    samples = load_dataset(path)
    total = len(samples)

    if total == 0:
        print("Dataset is empty!")
        return

    # Basic stats
    topics = Counter(s.get("topic", "") for s in samples)
    depths = Counter(s.get("depth", 0) for s in samples)
    agents_count = Counter(len(s.get("agents", [])) for s in samples)
    props_count = Counter(len(s.get("propositions", [])) for s in samples)
    formulas_count = Counter(len(s.get("formulas", [])) for s in samples)

    # Unique topics
    unique_topics = len(topics)

    print(f"Dataset Statistics:")
    print(f"  Total samples: {total}")
    print(f"  Unique topics: {unique_topics}")
    print()

    print("Depth distribution:")
    for depth, count in sorted(depths.items()):
        print(f"  Depth {depth}: {count} samples ({count/total*100:.1f}%)")
    print()

    print("Agents per sample:")
    for count, samples_with_count in sorted(agents_count.items()):
        print(f"  {count} agents: {samples_with_count} samples")
    print()

    print("Propositions per sample:")
    for count, samples_with_count in sorted(props_count.items()):
        print(f"  {count} propositions: {samples_with_count} samples")
    print()

    print("Formulas per sample:")
    for count, samples_with_count in sorted(formulas_count.items()):
        print(f"  {count} formulas: {samples_with_count} samples")
    print()

    print("Top 5 topics:")
    for topic, count in topics.most_common(5):
        print(f"  {topic[:60]}...: {count} samples")
    print()

    # Check for potential issues
    print("Data quality checks:")
    samples_with_meta = sum(1 for s in samples if s.get("metadata"))
    print(f"  Samples with metadata: {samples_with_meta}/{total}")

    # Verify agent names in formulas
    issues = 0
    for s in samples:
        text_agents = set()
        for line in s.get("text", "").split("\n"):
            if ":" in line:
                text_agents.add(line.split(":")[0].strip())

        for formula in s.get("formulas", []):
            # Extract agent from formula like B_Alex(p1)
            parts = formula.split("(")
            if len(parts) > 0:
                agent_part = parts[0].replace("B_", "").replace("¬", "").strip()
                if agent_part and agent_part not in text_agents:
                    issues += 1
                    break

    print(f"  Formulas with mismatched agents: {issues}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        path = "meta_opinions_dataset.jsonl"
    else:
        path = sys.argv[1]

    analyze_dataset(path)
