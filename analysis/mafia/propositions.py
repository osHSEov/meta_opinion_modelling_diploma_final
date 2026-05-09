from typing import Dict
from .types import EpistemicFact


class PropositionRegistry:

    def __init__(self):
        self.fact_to_prop: Dict[EpistemicFact, str] = {}
        self.prop_to_fact: Dict[str, EpistemicFact] = {}

    def get_or_create(self, fact: EpistemicFact) -> str:

        if fact not in self.fact_to_prop:

            idx = len(self.fact_to_prop) + 1
            prop = f"p{idx}"

            self.fact_to_prop[fact] = prop
            self.prop_to_fact[prop] = fact

        return self.fact_to_prop[fact]

    def propositions(self):
        return list(self.prop_to_fact.keys())

    def decode(self, prop: str):
        return self.prop_to_fact.get(prop)