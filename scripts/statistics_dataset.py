
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean


FORMULA_RE = re.compile(r"B_")
NEG_RE = re.compile(r"¬")


def load_jsonl(path):

    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def formula_depth(formula):

    return formula.count("B_")


def is_meta(formula):

    return formula_depth(formula) > 1


def compute_stats(samples):

    stats = {}

    stats["samples"] = len(samples)

    stats["avg_agents"] = mean(
        len(x["agents"])
        for x in samples
    )

    stats["avg_propositions"] = mean(
        len(x["propositions"])
        for x in samples
    )

    stats["avg_formulas"] = mean(
        len(x["formulas"])
        for x in samples
    )

    stats["avg_dialogue_chars"] = mean(
        len(x["text"])
        for x in samples
    )

    all_formulas = []

    for x in samples:
        all_formulas.extend(x["formulas"])

    stats["total_formulas"] = len(all_formulas)

    depths = [
        formula_depth(f)
        for f in all_formulas
    ]

    stats["avg_depth"] = mean(depths)
    stats["max_depth"] = max(depths)

    depth_distribution = Counter(depths)

    stats["depth_distribution"] = dict(depth_distribution)

    meta_count = sum(
        1 for f in all_formulas
        if is_meta(f)
    )

    stats["meta_belief_ratio"] = (
        meta_count / len(all_formulas)
    )

    negative_count = sum(
        1 for f in all_formulas
        if "¬" in f or "~" in f
    )

    stats["negative_ratio"] = (
        negative_count / len(all_formulas)
    )

    agent_counter = Counter()

    for x in samples:
        for a in x["agents"]:
            agent_counter[a] += 1

    stats["unique_agents"] = len(agent_counter)

    prop_lengths = []

    for x in samples:
        for p in x["propositions"]:
            prop_lengths.append(len(p))

    stats["avg_proposition_length"] = mean(prop_lengths)

    stats["formula_per_agent"] = mean(
        len(x["formulas"]) / len(x["agents"])
        for x in samples
    )

    return stats


def print_stats(stats):

    print()
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    for k, v in stats.items():
        print(f"{k}: {v}")


def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)

    args = parser.parse_args()

    samples = load_jsonl(args.input)

    stats = compute_stats(samples)

    print_stats(stats)


if __name__ == "__main__":
    main()