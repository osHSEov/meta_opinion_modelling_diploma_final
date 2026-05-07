from .ollama_client import OllamaClient
from .vllm_client import VLLMClient


def build_llm_client(config: dict):
    backend = config["backend"]

    if backend == "ollama":
        return OllamaClient(config)

    if backend == "vllm":
        return VLLMClient(config)

    raise ValueError(f"Unknown backend: {backend}")