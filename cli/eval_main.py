import argparse
import json
from evaluation.evaluator import evaluate_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on the synthetic meta‑opinions dataset")
    parser.add_argument("--dataset", default="meta_opinions_dataset.jsonl",
                        help="Path to the JSON‑L dataset")
    parser.add_argument("--model", required=True,
                        help="Model name to evaluate (e.g. gpt-oss:120b)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for the model")
    args = parser.parse_args()

    scores = evaluate_dataset(
        dataset_path=args.dataset,
        model_name=args.model,
        temperature=args.temperature,
    )
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
