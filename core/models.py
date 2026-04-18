from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SyntheticSample:
    id: str
    topic: str
    text: str
    agents: List[str]
    propositions: List[str]
    formulas: List[str]
    depth: int
    metadata: Dict
    
    def to_dict(self) -> Dict[str, any]:
        return asdict(self)