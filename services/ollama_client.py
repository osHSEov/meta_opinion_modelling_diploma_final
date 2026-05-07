import ollama
from typing import List, Dict, Any
from .base_llm_client import BaseLLMClient

class OllamaClient(BaseLLMClient):
    def __init__(self, config: dict):
        self.model = config["name"]
        self.temperature = config["temperature"]
        self.seed = config["seed"]
        #self.num_ctx = config["num_ctx"]

    def chat(
        self, messages: List[Dict[str, str]], seed_offset: int = 0
    ) -> Dict[str, Any]:
        return ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
                #"seed": self.seed + seed_offset,
                #"num_ctx": self.num_ctx,
            },
        )