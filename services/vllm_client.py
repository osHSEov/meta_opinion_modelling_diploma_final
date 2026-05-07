import requests
from typing import List, Dict, Any
from .base_llm_client import BaseLLMClient

class VLLMClient(BaseLLMClient):
    def __init__(self, config:dict):
        self.base_url = config["base_url"]
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 512)
    
    def chat(self, messages: List[Dict[str, str]], seed_offset: int = 0) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
             "max_tokens": self.max_tokens,
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=600,
        )
        
        response.raise_for_status()
        data = response.json()
        
        return {
            "message": {
                "content": data["choices"][0]["message"]["content"]
            }
        }