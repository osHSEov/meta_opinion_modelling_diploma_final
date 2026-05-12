import json
import os
from dataclasses import asdict

def save_jsonl(path, samples, append=False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")