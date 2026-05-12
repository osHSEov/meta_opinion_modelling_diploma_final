from typing import List, Set, Dict, Tuple
from core.ast_parser import parse_formula, Proposition, Not, Belief, Node
from kripke.model import KripkeModel

class KripkeBuilder:
    def __init__(self, agents: List[str], propositions: List[str], formulas: List[str]):
        
        self.agents = agents
        self.propositions = propositions
        self.asts = [parse_formula(f) for f in formulas]
        self.world_counter = 0
        self.worlds: Dict[int, Set[Node]] = {}
        self.relations: Dict[str, Set[Tuple[int, int]]] = {a: set() for a in agents}
        self.valuation: Dict[int, Set[int]] = {}
        self._build()

    def _new_world(self) -> int:
        w = self.world_counter
        self.world_counter += 1
        self.worlds[w] = set()
        self.valuation[w] = set()
        return w

    def _add_requirement(self, world: int, formula: Node):
        if formula in self.worlds[world]:
            return
        self.worlds[world].add(formula)

        if isinstance(formula, Proposition):
            idx = int(formula.name[1:])
            self.valuation[world].add(idx)

        elif isinstance(formula, Not):
            child = formula.child
            
            if isinstance(child, Proposition):
                idx = int(child.name[1:])
                self.valuation[world].discard(idx)
                
            elif isinstance(child, Belief):
                agent = child.agent
                sub = child.child
                v = self._new_world()
                self.relations[agent].add((world, v))
                self._add_requirement(v, Not(sub))
                
            else:
                self._add_requirement(world, child.child)

        elif isinstance(formula, Belief):
            agent = formula.agent
            sub = formula.child
            v = self._new_world()
            self.relations[agent].add((world, v))
            self._add_requirement(v, sub)

        else: # На всякий
            raise ValueError(f"Unsupported formula type: {type(formula)}")

    def _propagate_beliefs(self):
        changed = True
        while changed:
            changed = False
            for w, formulas in list(self.worlds.items()):
                for f in formulas:
                    if isinstance(f, Belief):
                        agent = f.agent
                        sub = f.child
                        for (u, v) in list(self.relations[agent]):
                            if u == w:
                                if sub not in self.worlds[v]:
                                    self._add_requirement(v, sub)
                                    changed = True

    def _build(self):
        root = self._new_world()
        for ast in self.asts:
            self._add_requirement(root, ast)

        self._propagate_beliefs()

        self.model = KripkeModel(
            worlds=list(self.worlds.keys()),
            agents=self.agents,
            propositions=self.propositions,
            relations=self.relations,
            valuation=self.valuation,
            requirements=self.worlds,      
            enforce_frame=True
        )