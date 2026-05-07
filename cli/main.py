import json
import os
from dataclasses import asdict
import yaml
from services.ollama_client import OllamaClient
from core.generator import MetaOpinionDatasetGenerator
from utils.io import save_jsonl
from services.llm_factory import build_llm_client
import random


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
    
    
    generation_config = {
        "max_retries": config["max_retries"],
        "max_depth_limit": 3,
    }
    
    generator = MetaOpinionDatasetGenerator(
        client,
        generation_config,
    )

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
    topics = generator.generate_topics(config["topics_pool_size"])
    print(f"Generated {len(topics)} topics")

    samples_buffer = []
    
    diversity_profiles = config["diversity_profiles"]
    profile_weights = config["profile_weights"]

    profile_names = list(profile_weights.keys())
    weights = list(profile_weights.values())

    generated = 0
    skipped = 0
    
    while generated < config["samples_total"]:
        topic = random.choice(topics)

        profile_name = random.choices(
            profile_names,
            weights=weights,
            k=1,
        )[0]
        
        profile = diversity_profiles[profile_name]

        sample = generator.generate_sample(
            topic=topic,
            num_agents=random.randint(*profile["num_agents"]),
            max_depth=random.randint(*profile["max_depth"]),
            min_props=profile["min_props"],
            max_props=profile["max_props"],
            style=profile["style"],
        )

        if sample is None:
            continue

        if sample.id in existing_ids:
            skipped += 1
            continue

        existing_ids.add(sample.id)

        samples_buffer.append(sample)

        generated += 1

        print(
            f"[{generated}/{config['samples_total']}] "
            f"{sample.topic} "
            f"({profile_name})"
        )

        if len(samples_buffer) >= 5:

            save_jsonl(
                output_path,
                samples_buffer,
                append=True,
            )

            samples_buffer = []

    if samples_buffer:
        save_jsonl(
            output_path,
            samples_buffer,
            append=True,
        )

    print(
        f"\nDone.\n"
        f"Generated: {generated}\n"
        f"Skipped duplicates: {skipped}"
    )
        

    
if __name__ == "__main__":
    main()
