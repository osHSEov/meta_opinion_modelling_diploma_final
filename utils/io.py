import json
from dataclasses import asdict

def save_jsonl(path, samples, append=False):
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")