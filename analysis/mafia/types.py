from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EpistemicFact:

    subject: str
    predicate: str
    obj: Optional[str] = None
    polarity: bool = True

    def symbol(self):

        if self.obj:
            base = f"{self.subject}_{self.predicate}_{self.obj}"
        else:
            base = f"{self.subject}_{self.predicate}"

        return base.lower()


@dataclass
class BeliefStatement:

    agent: str

    fact: Optional[EpistemicFact] = None

    confidence: float = 1.0

    nested_formula: Optional[str] = None

    source: str = "introspection"