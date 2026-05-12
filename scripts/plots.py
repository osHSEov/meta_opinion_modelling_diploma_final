# analysis/plots.py

import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt


def load_jsonl(path):

    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data

def plot_agents_distribution(samples, output_dir):
    
    agent_counts = [len(x["agents"]) for x in samples]
    
    plt.figure(figsize=(8, 5))
    plt.hist(agent_counts, bins=range(1, max(agent_counts)+2), align='left', rwidth=0.8)
    plt.xlabel("Количество агентов в диалоге")
    plt.ylabel("Количество диалогов")
    plt.title("Распределение числа участников")
    plt.xticks(range(1, max(agent_counts)+1))
    
    path = output_dir / "agents_distribution.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", path)

def formula_depth(formula):

    return formula.count("B_")


def is_meta(formula):

    return formula_depth(formula) > 1


def plot_depth_distribution(samples, output_dir):

    depths = []

    for x in samples:
        for f in x["formulas"]:
            depths.append(formula_depth(f))

    counter = Counter(depths)

    xs = sorted(counter.keys())
    ys = [counter[x] for x in xs]

    plt.figure(figsize=(8, 5))

    plt.bar(xs, ys)

    plt.xlabel("Belief Nesting Depth")
    plt.ylabel("Formula Count")
    plt.title("Distribution of Epistemic Nesting Depth")

    path = output_dir / "depth_distribution.png"

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", path)

def plot_meta_ratio(samples, output_dir):

    ordinary = 0
    meta = 0

    for x in samples:
        for f in x["formulas"]:

            if is_meta(f):
                meta += 1
            else:
                ordinary += 1

    plt.figure(figsize=(6, 6))

    plt.pie(
        [ordinary, meta],
        labels=["Ordinary beliefs", "Meta-beliefs"],
        autopct="%1.1f%%"
    )

    plt.title("Meta-Belief Ratio")

    path = output_dir / "meta_ratio.png"

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


def plot_negation_ratio(samples, output_dir):

    positive = 0
    negative = 0

    for x in samples:
        for f in x["formulas"]:

            if "¬" in f or "~" in f:
                negative += 1
            else:
                positive += 1

    plt.figure(figsize=(6, 5))

    plt.bar(
        ["Positive", "Negative"],
        [positive, negative]
    )

    plt.title("Belief Polarity Distribution")

    path = output_dir / "negation_ratio.png"

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", path)
    

def plot_formula_distribution(samples, output_dir):

    counts = [
        len(x["formulas"])
        for x in samples
    ]

    plt.figure(figsize=(8, 5))

    plt.hist(counts, bins=20)

    plt.xlabel("Formulas per Sample")
    plt.ylabel("Samples")
    plt.title("Formula Density Distribution")

    path = output_dir / "formula_distribution.png"

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


def plot_proposition_distribution(samples, output_dir):

    counts = [
        len(x["propositions"])
        for x in samples
    ]

    plt.figure(figsize=(8, 5))

    plt.hist(counts, bins=20)

    plt.xlabel("Propositions per Sample")
    plt.ylabel("Samples")
    plt.title("Semantic Proposition Distribution")

    path = output_dir / "proposition_distribution.png"

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)

    parser.add_argument(
        "--output-dir",
        default="./plots"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    samples = load_jsonl(args.input)

    plot_depth_distribution(samples, output_dir)

    plot_meta_ratio(samples, output_dir)

    plot_negation_ratio(samples, output_dir)

    plot_formula_distribution(samples, output_dir)

    plot_proposition_distribution(samples, output_dir)
    
    plot_agents_distribution(samples, output_dir)


if __name__ == "__main__":
    main()