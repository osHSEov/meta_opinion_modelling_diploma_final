from dataclasses import dataclass
from typing import Union, List
import re
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Proposition:
    name: str

@dataclass(frozen=True)
class Not:
    child: "Node"

@dataclass(frozen=True)
class Belief:
    agent: str
    child: "Node"


Node = Union[Proposition, Not, Belief]

def parse_formula(s: str) -> Node:
    s = s.replace(" ", "")

    if re.fullmatch(r"p\d+", s):
        return Proposition(s)

    if s.startswith("¬"):
        return Not(parse_formula(s[1:]))

    match = re.match(r"B_([A-Za-z]+)\((.*)\)", s)
    if match:
        agent = match.group(1)
        inner = match.group(2)
        return Belief(agent, parse_formula(inner))

    raise ValueError(f"Invalid formula: {s}")

def compute_depth(node: Node) -> int:
    if isinstance(node, Proposition):
        return 0
    if isinstance(node, Not):
        return compute_depth(node.child)
    if isinstance(node, Belief):
        return 1 + compute_depth(node.child)

def max_depth(formulas: List[str]) -> int:
    if not formulas:
        return 0
    depths = []
    for f in formulas:
        try:
            depths.append(compute_depth(parse_formula(f)))
        except Exception:
            continue
    return max(depths) if depths else 0

def extract_components(node: Node):
    agents = []
    props = []
    negations = 0

    def visit(n):
        nonlocal negations

        if isinstance(n, Proposition):
            props.append(n.name)

        elif isinstance(n, Not):
            negations += 1
            visit(n.child)

        elif isinstance(n, Belief):
            agents.append(n.agent)
            visit(n.child)

    visit(node)

    return {
        "agents": agents,
        "props": props,
        "negations": negations
    }

def formula_similarity_heuristic(a: Node, b: Node) -> float:
    ca = extract_components(a)
    cb = extract_components(b)

    score = 0
    total = 3

    if ca["agents"] == cb["agents"]:
        score += 1

    if ca["props"] == cb["props"]:
        score += 1

    if ca["negations"] == cb["negations"]:
        score += 1

    return score / total


def formula_tree_similarity(a: Node, b: Node) -> float:
    if type(a) == type(b):
        if isinstance(a, Proposition):
            return 1.0 if a.name == b.name else 0.0

        elif isinstance(a, Not):
            return 0.9 * formula_tree_similarity(a.child, b.child)

        elif isinstance(a, Belief):
            if a.agent != b.agent:
                return 0.0
            return formula_tree_similarity(a.child, b.child)

    return 0.0

def formula_similarity(a, b, mode="tree"):
    if mode == "tree":
        return formula_tree_similarity(a, b)
    elif mode == "heuristic":
        return formula_similarity_heuristic(a, b)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    
def safe_parse_list(formulas: List[str]):
    asts = []
    for f in formulas:
        try:
            asts.append(parse_formula(f))
        except Exception:
            continue
    return asts

def match_formulas(gt_list: List[str], pred_list: List[str], mode="tree"):
    gt_asts = safe_parse_list(gt_list)
    pred_asts = safe_parse_list(pred_list)

    if not gt_asts or not pred_asts:
        return 0.0, 0.0, 0.0

    sim_matrix = np.zeros((len(gt_asts), len(pred_asts)))

    for i, g in enumerate(gt_asts):
        for j, p in enumerate(pred_asts):
            sim_matrix[i, j] = formula_similarity(g, p, mode=mode)

    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    total_score = sim_matrix[row_ind, col_ind].sum()

    precision = total_score / len(pred_list) if pred_list else 0.0
    recall = total_score / len(gt_list) if gt_list else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1