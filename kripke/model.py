from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
import itertools

@dataclass
class KripkeModel:
    worlds: List[int]                       
    agents: List[str]                       
    propositions: List[str]                 
    relations: Dict[str, Set[tuple]]       
    valuation: Dict[int, Set[int]]         

    def __post_init__(self):
        self._enforce_s5()

    def _enforce_s5(self):
        for agent in self.agents:
            rel = self.relations.get(agent, set())
            
            for w in self.worlds:
                rel.add((w, w))
            
            symm = set((v, u) for (u, v) in rel)
            rel.update(symm)
            
            closure = set(rel)
            while True:
                new_pairs = set((u, w) for (u, v) in closure for (x, w) in closure if v == x)
                if new_pairs.issubset(closure):
                    break
                closure.update(new_pairs)
            self.relations[agent] = closure

    def world_entails(self, world: int, prop_index: int) -> bool:
        return prop_index in self.valuation.get(world, set())

    def check_formula(self, world: int, ast_node: 'Node') -> bool:
        from core.ast_parser import Proposition, Not, Belief  
        if isinstance(ast_node, Proposition):
            idx = int(ast_node.name[1:]) 
            return self.world_entails(world, idx)
        elif isinstance(ast_node, Not):
            return not self.check_formula(world, ast_node.child)
        elif isinstance(ast_node, Belief):
            agent = ast_node.agent
            child = ast_node.child
            
            accessible = {v for (u, v) in self.relations[agent] if u == world}
            return all(self.check_formula(v, child) for v in accessible)
        else:
            raise ValueError(f"Unknown AST node: {ast_node}")