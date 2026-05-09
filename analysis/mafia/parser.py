import json
from pathlib import Path


class MafiaGameParser:

    def __init__(self, logs_dir, game_id):

        self.logs_dir = Path(logs_dir)
        self.game_id = game_id

        self.game_path = self.logs_dir / game_id / "game.jsonl"
        self.introspection_path = self.logs_dir / game_id / "introspection.jsonl"

        self.game_events = []
        self.introspection = []

    def load(self):

        with open(self.game_path, "r", encoding="utf-8") as f:
            for line in f:
                self.game_events.append(json.loads(line))

        with open(self.introspection_path, "r", encoding="utf-8") as f:
            for line in f:
                self.introspection.append(json.loads(line))

    def players(self):

        for e in self.game_events:
            if e["kind"] == "setup":
                return [p["name"] for p in e["players"]]

        return []

    def player_roles(self):

        for e in self.game_events:
            if e["kind"] == "setup":
                return {
                    p["name"]: p["role"]
                    for p in e["players"]
                }

        return {}

    def grouped_introspection(self):

        grouped = {}

        for r in self.introspection:

            key = (
                r.get("round", 0),
                r.get("public_msg_seq", -1)
            )

            grouped.setdefault(key, []).append(r)

        return grouped
    
    def alive_players(self, round_num):

        alive = set(self.players())

        for e in self.game_events:

            if e["kind"] == "day_eliminate":

                eliminated = e.get("eliminated")

                eliminate_round = e.get("round", 0)

                if eliminate_round < round_num:
                    alive.discard(eliminated)

            if e["kind"] == "night_kill":

                killed = e.get("killed")

                kill_round = e.get("round", 0)

                if kill_round < round_num:
                    alive.discard(killed)

        return sorted(alive)
    