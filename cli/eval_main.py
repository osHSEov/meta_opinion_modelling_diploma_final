import argparse
import json
from evaluation.evaluator import evaluate_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on the synthetic meta‑opinions dataset")
    parser.add_argument("--dataset", default="meta_opinions_dataset.jsonl",
                        help="Path to the JSON‑L dataset")
    parser.add_argument("--model", required=True,
                        help="Model name to evaluate (e.g. gpt-oss:120b)")
    parser.add_argument("--backend", choices=["ollama", "vllm"], default="ollama",
                        help="Backend to use for inference")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Base URL for vLLM server (only used if backend=vllm)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for the model")
    parser.add_argument("--sample_random", type=int, default=None,
                    help="Randomly sample N examples from dataset")
    
    args = parser.parse_args()

    scores = evaluate_dataset(
        dataset_path=args.dataset,
        model_name=args.model,
        backend=args.backend,
        base_url=args.base_url,
        temperature=args.temperature,
        sample_random=args.sample_random,
    )
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
