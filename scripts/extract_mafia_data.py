#!/usr/bin/env python3
"""Extract meta-opinion data from mafia game introspection logs."""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def extract_meta_opinions(intro_data: List[Dict], game_data: List[Dict]) -> List[Dict]:
    """Extract meta-opinion samples from game data."""
    samples = []

    # Build player info from game.jsonl setup
    player_info = {}
    for event in game_data:
        if event.get("kind") == "setup":
            for p in event.get("players", []):
                player_info[p["name"]] = p

    # Group introspection events by game_id and player
    introspection_by_player = {}
    for event in intro_data:
        game_id = event.get("game_id")
        player_name = event.get("player_name")
        key = (game_id, player_name)
        if key not in introspection_by_player:
            introspection_by_player[key] = []
        introspection_by_player[key].append(event)

    for (game_id, player_name), events in introspection_by_player.items():
        player = player_info.get(player_name, {})
        player_role = player.get("role", "Unknown")

        # Look at social_map probes (these contain meta-beliefs)
        for event in events:
            if event.get("probe_id") == "social_map":
                # Extract what player thinks others think of them
                social_map = event.get("answer_parsed", {})
                toward_me = social_map.get("toward_me", [])

                if not toward_me:
                    continue

                # Build propositions from this
                propositions = []
                for entry in toward_me:
                    attitude = entry.get("attitude", "neutral")
                    prop = f"{player_name} believes {entry['player']} {attitude}s them"
                    if prop not in propositions:
                        propositions.append(prop)

                # Build formulas - direct beliefs about others' attitudes toward player
                # This is the meta-level: player holds beliefs about others' attitudes
                formulas = []
                agents = set([player_name])

                for entry in toward_me:
                    other = entry.get("player")
                    agents.add(other)
                    attitude = entry.get("attitude", "neutral")

                    # B_player(attitude_toward_me) = player believes others have this attitude toward them
                    # Since this is about player's meta-belief (belief about others' beliefs/attitudes),
                    # we model it as depth 1: player holds belief about others' state
                    formula = f"B_{player_name}({attitude}_{other})"
                    formulas.append(formula)

                # Create a topic from the game context
                topic = f"Mafia game: {game_id[:8]}... (social_map introspection)"

                # Create sample ID
                content = json.dumps({"player": player_name, "game": game_id})
                sample_id = hashlib.md5(content.encode()).hexdigest()[:12]

                sample = {
                    "id": sample_id,
                    "topic": topic,
                    "text": f"{player_name}: Analysis of how others see me:\n" + "\n".join(
                        f"  - {e['player']}: {e['attitude']} (confidence: {e['confidence']}%)" + (f" - {e.get('reason', '')}" if e.get('reason') else "")
                        for e in toward_me
                    ),
                    "agents": sorted(list(agents)),
                    "propositions": propositions[:5],  # Limit to 5
                    "formulas": formulas[:10],  # Limit to 10
                    "depth": 1,  # Direct meta-beliefs about others' attitudes
                    "metadata": {
                        "game_id": game_id,
                        "player": player_name,
                        "player_role": player_role,
                        "probe_type": "social_map",
                        "probe_timestamp": event.get("timestamp"),
                    }
                }

                samples.append(sample)

    return samples


def main():
    intro_path = "interospection.jsonl"
    game_path = "game.jsonl"
    output_path = "meta_opinions_dataset_mafia.jsonl"

    if not Path(intro_path).exists():
        print(f"Error: {intro_path} not found")
        return

    print(f"Loading {intro_path}.?.")
    intro_data = load_jsonl(intro_path)
    print(f"Loaded {len(intro_data)} introspection events")

    print(f"Loading {game_path}.?.")
    game_data = load_jsonl(game_path)
    print(f"Loaded {len(game_data)} game events")

    print("Extracting meta-opinion samples...")
    samples = extract_meta_opinions(intro_data, game_data)
    print(f"Extracted {len(samples)} samples")

    if not samples:
        print("Warning: No samples extracted!")
        return

    # Save to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved to {output_path}")

    # Show sample
    print("\nFirst sample:")
    print(json.dumps(samples[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
