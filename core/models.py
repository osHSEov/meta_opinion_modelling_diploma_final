from dataclasses import dataclass
from typing import List, Dict
import re


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


def validate_formulas_match_text(formulas: List[str], text: str) -> bool:
    """Check that formulas can be reasonably inferred from the text."""
    text_lower = text.lower()
    formula_agents = set()
    for formula in formulas:
        match = re.search(r'B_([A-Za-z]+)', formula)
        if match:
            formula_agents.add(match.group(1).lower())

    text_agents = set()
    for line in text.split('\n'):
        if ':' in line:
            agent = line.split(':')[0].strip().lower()
            if agent:
                text_agents.add(agent)

    return len(formula_agents - text_agents) == 0


def count_nesting_depth(formula: str) -> int:
    """Count the maximum nesting depth of B-operators in a formula."""
    max_depth = 0
    current_depth = 0
    for char in formula:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1
    return max_depth


def validate_formula_depth(formulas: List[str], expected_depth: int) -> bool:
    """Validate that formulas have the expected nesting depth."""
    for formula in formulas:
        depth = count_nesting_depth(formula)
        if depth != expected_depth:
            return False
    return True
