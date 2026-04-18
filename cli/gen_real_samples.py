#!/usr/bin/env python3
"""Generate realistic-style dialogues for the meta-opinions benchmark."""

import json
import os
from dataclasses import asdict
import yaml
from services.ollama_client import OllamaClient
from core.real_generator import RealStyleDatasetGenerator


def save_jsonl(path, samples, append=False):
    """Save samples to JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")


def load_existing_ids(path: str) -> set:
    """Load existing sample IDs to avoid duplicates."""
    existing_ids = set()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if "id" in data:
                            existing_ids.add(data["id"])
        except (json.JSONDecodeError, IOError):
            pass
    return existing_ids


def main():
    with open("config/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    client = OllamaClient(config["model"])
    generator = RealStyleDatasetGenerator(client, config["real_generation"])

    output_path = "meta_opinions_dataset_real.jsonl"
    append_mode = config["output"].get("append", True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Load existing IDs if appending
    if append_mode:
        existing_ids = load_existing_ids(output_path)
        print(f"Found {len(existing_ids)} existing samples in real dataset")
    else:
        existing_ids = set()

    print("Generating topics...")
    topics = generator.generate_topics(config["generation"].get("num_topics", 50))
    print(f"Generated {len(topics)} topics")

    count = 0
    skipped = 0
    samples_buffer = []

    for i, topic in enumerate(topics, 1):
        print(f"[{i:02d}/{len(topics)}] Generating samples for: {topic}")

        for j in range(config["generation"]["samples_per_topic"]):
            sample = generator.generate_sample(topic)
            if sample:
                if sample.id in existing_ids:
                    skipped += 1
                    continue
                samples_buffer.append(sample)
                existing_ids.add(sample.id)
                count += 1

        # Save after each topic
        if samples_buffer:
            save_jsonl(output_path, samples_buffer, append=True)
            samples_buffer = []

    print(f"Real-style dataset: {count} samples → {output_path}")
    print(f"Skipped {skipped} duplicates")


if __name__ == "__main__":
    main()
