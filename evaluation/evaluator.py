import json
import re
from typing import List, Dict, Tuple

from services.ollama_client import OllamaClient
from core.parser import parse_response
from core.prompts import build_system_prompt, build_extraction_prompt
from core.models import SyntheticSample

from tqdm import tqdm
import random

from sentence_transformers import SentenceTransformer, util


def normalize_formula(f: str) -> str:
    f = f.replace(" ", "")
    f = f.replace("¬", "NOT_")
    return f


def normalize_list(lst: List[str]) -> List[str]:
    return [x.strip().lower() for x in lst]


def load_dataset(path: str) -> List[SyntheticSample]:
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
) -> Tuple[Dict, Dict, bool]:
    
    prompt = build_extraction_prompt(sample.text)

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
            return sample.to_dict(), parsed, True

    return sample.to_dict(), {}, False


def score_lists(gt: List[str], pred: List[str]) -> Tuple[float, float, float]:
    gt_set, pred_set = set(gt), set(pred)

    if not gt_set and not pred_set:
        return 1.0, 1.0, 1.0

    tp = len(gt_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1

def fuzzy_score_propositions(
    gt_props: List[str],
    pred_props: List[str],
    model: SentenceTransformer,
    threshold: float = 0.7
) -> Tuple[float, float, float]:
    
    if not gt_props and not pred_props:
        return 1.0, 1.0, 1.0
    if not gt_props or not pred_props:
        return 0.0, 0.0, 0.0

    gt_emb = model.encode(gt_props, convert_to_tensor=True)
    pred_emb = model.encode(pred_props, convert_to_tensor=True)

    sim_matrix = util.cos_sim(gt_emb, pred_emb).cpu().numpy()

    matched_gt = set()
    matched_pred = set()

    for i in range(len(gt_props)):
        best_j = -1
        best_sim = 0.0
        for j in range(len(pred_props)):
            if j in matched_pred:
                continue
            sim = sim_matrix[i, j]
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_sim >= threshold:
            matched_gt.add(i)
            matched_pred.add(best_j)

    tp = len(matched_gt)
    precision = tp / len(pred_props)
    recall = tp / len(gt_props)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1

def evaluate_dataset(
    dataset_path: str,
    model_name: str,
    temperature: float = 0.0,
    max_retries: int = 2,
    sample_random: int = None,
    fuzzy_threshold: float = 0.7
) -> Dict[str, float]:

    client = OllamaClient({
        "name": model_name,
        "temperature": temperature,
        "seed": 42
    })

    samples = load_dataset(dataset_path)
    if sample_random is not None:
        samples = random.sample(samples, min(sample_random, len(samples)))
    total = len(samples)
    
    st_model = SentenceTransformer('all-MiniLM-L6-v2')

    agg = {
        "agents_f1": 0.0,
        "props_f1": 0.0,
        "formulas_f1": 0.0,
        "depth_mae": 0.0,
        "depth_acc": 0.0,
        "parse_success_rate": 0.0,
    }

    for s in tqdm(samples, desc="Evaluating"):
        gt, pred, success = evaluate_sample(client, s, max_retries=max_retries)

        if success:
            agg["parse_success_rate"] += 1

        
        gt_agents = normalize_list(gt["agents"])
        pred_agents = normalize_list(pred.get("agents", []))
        _, _, f = score_lists(gt_agents, pred_agents)
        agg["agents_f1"] += f

        
        gt_props = normalize_list(gt["propositions"])
        pred_props = normalize_list(pred.get("propositions", []))
        if success and pred_props:
            _, _, f_props = fuzzy_score_propositions(gt_props, pred_props, st_model, threshold=fuzzy_threshold)
        else:
            f_props = 0.0
        agg["props_f1"] += f_props

        
        gt_formulas = [normalize_formula(f) for f in gt["formulas"]]
        pred_formulas = [normalize_formula(f) for f in pred.get("formulas", [])]
        _, _, f = score_lists(gt_formulas, pred_formulas)
        agg["formulas_f1"] += f

        
        pred_depth = pred.get("depth")
        if isinstance(pred_depth, int):
            agg["depth_mae"] += abs(gt["depth"] - pred_depth)
            if gt["depth"] == pred_depth:
                agg["depth_acc"] += 1
        else:
            agg["depth_mae"] += gt["depth"] 

    
    for k in ["agents_f1", "props_f1", "formulas_f1", "depth_mae", "depth_acc", "parse_success_rate"]:
        agg[k] = agg[k] / total if total else 0.0

    return agg