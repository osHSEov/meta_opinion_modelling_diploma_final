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
        if not self.enforce_frame or self.requirements is None:
            return

        changed = True
        
        while changed:
            before_rel = {a: set(r) for a, r in self.relations.items()}
            before_req = {w: set(f) for w, f in self.requirements.items()}

            self._enforce_kd45()
            self._saturate_beliefs()

            if before_rel == self.relations and before_req == self.requirements:
                changed = False
                
        self.worlds = list(self.requirements.keys())

    def _enforce_kd45(self):
        
        for agent in self.agents:
            
            rel = set(self.relations.get(agent, set()))
            all_worlds = set(self.worlds)

            while True:
                changed = False
                
                while True:
                    new_rel = set(rel)
                    # Гарантируем транзитивность: если (u,v) и (v,w) -> (u,w)
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

                # Серийность: из каждого мира должно идти хоть что-то
                worlds_with_outgoing = {u for (u, v) in rel}
                worlds_needing_edges = all_worlds - worlds_with_outgoing

                if worlds_needing_edges:
                    reachable_worlds = {v for (u,v) in rel}
                    if reachable_worlds:
                        target = next(iter(reachable_worlds))
                    else:
                        target = 0 if 0 in all_worlds else next(iter(all_worlds))
                    
                    for w in worlds_needing_edges:
                        rel.add((w, target))
                    
                    changed = True
                
                if not changed:
                    break

            self.relations[agent] = rel

    def _saturate_beliefs(self):
        changed = True
        while changed:
            changed = False
            for w, formulas in list(self.requirements.items()):
                for f in formulas:
                    if isinstance(f, Belief):
                        
                        agent = f.agent
                        sub = f.child
                        
                        targets = {v for (u, v) in self.relations[agent] if u == w}
                        
                        for v in targets:
                            if self._add_requirement_fixed(v, sub):
                                changed = True
                                
    def _add_requirement_fixed(self, world, formula):
        
        if formula in self.requirements[world]:
            return False
        
        self.requirements[world].add(formula)
         
        if isinstance(formula, Proposition):
            idx = int(formula.name[1:])
            self.valuation.setdefault(world, set()).add(idx)
            
        elif isinstance(formula, Not):
            
            child = formula.child
            if isinstance(child, Proposition):
                idx = int(child.name[1:])
                self.valuation.setdefault(world, set()).discard(idx)

            elif isinstance(child, Belief):
                agent = child.agent
                sub = child.child

                targets = {v for (u, v) in self.relations[agent] if u == world}

                if not targets: # При необходимоти создаем заглушку своего рода (новый мир как свидетель)
                    
                    v = max(self.worlds) + 1
                    self.worlds.append(v)
                    self.relations[agent].add((world, v))
                    self.requirements[v] = set()
                    self.valuation[v] = set()
                    targets = {v}

                for v in targets:
                    self._add_requirement_fixed(v, Not(sub))

            else:
                return self._add_requirement_fixed(world, child.child)

        elif isinstance(formula, Belief):
            
            agent = formula.agent
            sub = formula.child

            targets = {v for (u, v) in self.relations[agent] if u == world}

            if not targets:
                return False
            
            changed = False

            for v in targets:
                self._add_requirement_fixed(v, sub)
        
        return True

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
                return True # По дефолту если некуда идти, то утверждение верно
            return all(self.check_formula(v, child) for v in accessible)

        else:
            raise ValueError(f"Unknown AST node: {ast_node}")