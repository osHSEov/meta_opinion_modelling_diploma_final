from dataclasses import dataclass
from typing import List, Set, Dict

@dataclass
class KripkeModel:
    worlds: List[int]
    agents: List[str]
    propositions: List[str]
    relations: Dict[str, Set[tuple]]
    valuation: Dict[int, Set[int]]
    enforce_frame: bool = True  

    def __post_init__(self):
        if self.enforce_frame:
            self._enforce_kd45()

    def _enforce_kd45(self):
        for agent in self.agents:
            rel = set(self.relations.get(agent, set()))
            all_worlds = set(self.worlds)

            while True:
                new_rel = set(rel)

                for (u, v) in rel:
                    for (x, w) in rel:
                        if v == x:
                            new_rel.add((u, w))

                for (u, v) in rel:
                    for (x, y) in rel:
                        if u == x and v != y:
                            new_rel.add((v, y))

                if new_rel == rel:
                    break
                rel = new_rel

            worlds_with_outgoing = {u for (u, v) in rel}
            worlds_needing_edges = all_worlds - worlds_with_outgoing

            if worlds_needing_edges and all_worlds:
                target = next(iter(all_worlds))
                for w in worlds_needing_edges:
                    rel.add((w, target))

            self.relations[agent] = rel

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

            if not accessible:
                return True

            return all(self.check_formula(v, child) for v in accessible)

        else:
            raise ValueError(f"Unknown AST node: {ast_node}")