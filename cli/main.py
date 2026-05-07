import json
import os
from dataclasses import asdict
import yaml
from services.ollama_client import OllamaClient
from core.generator import MetaOpinionDatasetGenerator
from utils.io import save_jsonl


def load_existing_ids(path: str) -> set:
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


def save_checkpoint(path: str, existing_ids: set, count: int, current_topic_idx: int):
    checkpoint_path = path + ".checkpoint"
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump({
            "ids": list(existing_ids),
            "count": count,
            "topic_idx": current_topic_idx
        }, f)


def load_checkpoint(path: str):
    checkpoint_path = path + ".checkpoint"
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("ids", [])), data.get("count", 0), data.get("topic_idx", 0)
        except (json.JSONDecodeError, IOError):
            pass
    return set(), 0, 0


def cleanup_checkpoint(path: str):
    checkpoint_path = path + ".checkpoint"
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


def main():
    with open("config/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    client = build_llm_client(config["model"])
    generator = MetaOpinionDatasetGenerator(client, config["generation"])

    output_path = config["output"]["file"]
    append_mode = config["output"].get("append", False)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    existing_ids, saved_count, start_topic_idx = load_checkpoint(output_path)

    if append_mode:
        existing_ids = load_existing_ids(output_path)
        print(f"Append mode: Found {len(existing_ids)} existing samples")
    elif start_topic_idx > 0:
        print(f"Resuming from checkpoint: topic {start_topic_idx}, {saved_count} samples")

    print("Generating topics...")
    topics = generator.generate_topics(config["generation"].get("num_topics", 20))
    print(f"Generated {len(topics)} topics")

    count = saved_count
    skipped = 0
    samples_buffer = []

    for i, topic in enumerate(topics, 1):
        if i < start_topic_idx:
            continue

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

                if count % 5 == 0:
                    save_checkpoint(output_path, existing_ids, count, i)

        if samples_buffer:
            first_save = count == len(samples_buffer)
            save_jsonl(output_path, samples_buffer, append=not first_save)
            samples_buffer = []

        save_checkpoint(output_path, existing_ids, count, i)

    cleanup_checkpoint(output_path)
    print(f"Dataset update: {count} total samples, {skipped} duplicates skipped → {output_path}")


if __name__ == "__main__":
    main()
