import json
from pathlib import Path
from typing import List, Dict, Tuple

from services.ollama_client import OllamaClient
from core.parser import parse_response
from core.prompts import build_system_prompt, build_user_prompt
from core.models import SyntheticSample


def load_dataset(path: str) -> List[SyntheticSample]:
    """Read the JSON‑L file and return a list of SyntheticSample objects."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            samples.append(SyntheticSample(**data))
    return samples


def evaluate_sample(
    client: OllamaClient,
    sample: SyntheticSample,
    max_retries: int = 2,
) -> Tuple[Dict, Dict]:
    """Prompt the model with the raw text and return (ground_truth, model_output)."""
    prompt = build_user_prompt(
        topic=sample.topic,
        max_depth=sample.depth,
        num_agents=len(sample.agents),
        min_props=len(sample.propositions),
        max_props=len(sample.propositions),
    )

    for attempt in range(max_retries):
        resp = client.chat(
            [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            seed_offset=attempt,
        )
        parsed = parse_response(resp["message"]["content"])
        if parsed and all(k in parsed for k in ("agents", "propositions", "formulas", "depth")):
            return sample.to_dict(), parsed
    # If parsing never succeeded, return empty dict for model side
    return sample.to_dict(), {}


def score_lists(gt: List[str], pred: List[str]) -> Tuple[float, float, float]:
    """Precision / recall / F1 for unordered string lists."""
    gt_set, pred_set = set(gt), set(pred)
    if not gt_set and not pred_set:
        return 1.0, 1.0, 1.0
    tp = len(gt_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate_dataset(
    dataset_path: str,
    model_name: str,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> Dict[str, float]:
    """Run the whole dataset through the model and aggregate scores."""
    client = OllamaClient({"name": model_name, "temperature": temperature, "seed": 42})
    samples = load_dataset(dataset_path)
    total = len(samples)
    agg = {
        "agents_f1": 0.0,
        "props_f1": 0.0,
        "formulas_f1": 0.0,
        "depth_mae": 0.0,
    }
    for s in samples:
        gt, pred = evaluate_sample(client, s, max_retries=max_retries)
        # agents
        _, _, f = score_lists(gt["agents"], pred.get("agents", []))
        agg["agents_f1"] += f
        # propositions
        _, _, f = score_lists(gt["propositions"], pred.get("propositions", []))
        agg["props_f1"] += f
        # formulas
        _, _, f = score_lists(gt["formulas"], pred.get("formulas", []))
        agg["formulas_f1"] += f
        # depth absolute error
        depth_err = abs(gt["depth"] - pred.get("depth", -1))
        agg["depth_mae"] += depth_err
    # average
    for k in agg:
        agg[k] = agg[k] / total if total else 0.0
    return agg
