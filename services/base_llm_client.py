from typing import List, Dict, Any
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict[str,str]], seed_offest: int = 0) -> Dict[str, Any]:
        pass