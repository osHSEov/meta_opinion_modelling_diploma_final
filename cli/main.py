import yaml
from services.ollama_client import OllamaClient
from core.generator import MetaOpinionDatasetGenerator
from utils.io import save_jsonl
import random

def main():
    with open("config/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    client = OllamaClient(config["model"])
    generator = MetaOpinionDatasetGenerator(client, config["generation"])

    print("Generating topics...")
    topics = generator.generate_topics(config["generation"].get("num_topics", 20))
    print(f"Generated {len(topics)} topics")

    samples = []
    total = len(topics) * config["generation"]["samples_per_topic"]

    for i, topic in enumerate(topics, 1):
        print(f"[{i:02d}/{len(topics)}] Generating samples for: {topic}")
        for _ in range(config["generation"]["samples_per_topic"]):
            sample = generator.generate_sample(topic)
            if sample:
                samples.append(sample)

    output_path = config["output"]["file"]
    save_jsonl(output_path, samples, append=config["output"].get("append", False))
    print(f"Dataset saved: {len(samples)} samples → {output_path}")


if __name__ == "__main__":
    main()