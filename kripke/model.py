from dataclasses import dataclass
from typing import List, Set, Dict, Optional
from core.ast_parser import Proposition, Not, Belief, Node

@dataclass
class KripkeModel:
    worlds: List[int]
    agents: List[str]
    propositions: List[str]
    relations: Dict[str, Set[tuple]]
    valuation: Dict[int, Set[int]]
    requirements: Optional[Dict[int, Set[Node]]] = None
    enforce_frame: bool = True

    def __post_init__(self):
        if self.enforce_frame:
            self._enforce_kd45()
        if self.requirements is not None:
            self._saturate_beliefs()

    def _enforce_kd45(self):
        """Добавляет транзитивность, евклидовость и серийность."""
        for agent in self.agents:
            rel = set(self.relations.get(agent, set()))
            all_worlds = set(self.worlds)

            # Транзитивность и евклидовость до фиксированной точки
            while True:
                new_rel = set(rel)

                # Транзитивность: (u,v) и (v,w) -> (u,w)
                for (u, v) in rel:
                    for (x, w) in rel:
                        if v == x:
                            new_rel.add((u, w))

                # Евклидовость: (u,v) и (u,w) -> (v,w)
                for (u, v) in rel:
                    for (x, y) in rel:
                        if u == x and v != y:
                            new_rel.add((v, y))

                if new_rel == rel:
                    break
                rel = new_rel

            # Серийность: у каждого мира должна быть хотя бы одна исходящая дуга
            worlds_with_outgoing = {u for (u, v) in rel}
            worlds_needing_edges = all_worlds - worlds_with_outgoing

            if worlds_needing_edges and all_worlds:
                # Если есть хоть один мир с исходящей дугой, цепляемся к нему,
                # иначе создаём петлю на первом попавшемся мире.
                if worlds_with_outgoing:
                    target = next(iter(worlds_with_outgoing))
                else:
                    target = next(iter(all_worlds))
                for w in worlds_needing_edges:
                    rel.add((w, target))

            self.relations[agent] = rel

    def _saturate_beliefs(self):
        """
        После KD45-замыкания гарантирует, что все Belief-формулы остаются истинными.
        Для каждого мира w и каждой формулы B_a φ из требований этого мира,
        добавляем φ во все миры v, достижимые из w по агенту a.
        """
        changed = True
        while changed:
            changed = False
            for w, formulas in self.requirements.items():
                for f in formulas:
                    if isinstance(f, Belief):
                        agent = f.agent
                        sub = f.child
                        for (u, v) in self.relations[agent]:
                            if u == w:
                                reqs = self.requirements.setdefault(v, set())
                                if sub not in reqs:
                                    reqs.add(sub)
                                    # Обновляем оценку для пропозициональных подформул
                                    if isinstance(sub, Proposition):
                                        idx = int(sub.name[1:])
                                        self.valuation.setdefault(v, set()).add(idx)
                                    elif isinstance(sub, Not) and isinstance(sub.child, Proposition):
                                        idx = int(sub.child.name[1:])
                                        self.valuation.setdefault(v, set()).discard(idx)
                                    changed = True
                            # Вложенные Belief обрабатываются рекурсивно благодаря changed

    def world_entails(self, world: int, prop_index: int) -> bool:
        return prop_index in self.valuation.get(world, set())

    def check_formula(self, world: int, ast_node: Node) -> bool:
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
                return True   # vacuous truth
            return all(self.check_formula(v, child) for v in accessible)

        else:
            raise ValueError(f"Unknown AST node: {ast_node}")