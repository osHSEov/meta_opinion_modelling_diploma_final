from collections import defaultdict
from typing import List, Set, Dict, Tuple
from kripke.model import KripkeModel

def compute_bisimulation_quotient(model: KripkeModel) -> KripkeModel:
   
    worlds = model.worlds
    agents = model.agents
    partition = []
    val_to_block = {}
    for w in worlds:
        val = frozenset(model.valuation.get(w, set()))
        if val not in val_to_block:
            block_id = len(partition)
            val_to_block[val] = block_id
            partition.append({w})
        else:
            partition[val_to_block[val]].add(w)

    changed = True
    while changed:
        changed = False
        new_partition = []
        for block in partition:
           
            sig_to_worlds = defaultdict(set)
            for w in block:
                sig = []
                for a in agents:
                    reachable_blocks = set()
                    for (u, v) in model.relations[a]:
                        if u == w:
                            for idx, blk in enumerate(partition):
                                if v in blk:
                                    reachable_blocks.add(idx)
                                    break
                    sig.append(frozenset(reachable_blocks))
                sig = tuple(sig)
                sig_to_worlds[sig].add(w)
            if len(sig_to_worlds) > 1:
                changed = True
                for worlds_subset in sig_to_worlds.values():
                    new_partition.append(worlds_subset)
            else:
                new_partition.append(block)
        partition = new_partition

    block_to_new_world = {}
    new_worlds = []
    new_valuation = {}
    for idx, block in enumerate(partition):
        block_to_new_world[frozenset(block)] = idx
        new_worlds.append(idx)
        rep = next(iter(block))
        new_valuation[idx] = model.valuation.get(rep, set()).copy()

    new_relations = {a: set() for a in agents}
    for a in agents:
        for (u, v) in model.relations[a]:
            for block_u in partition:
                if u in block_u:
                    bu = frozenset(block_u)
                    break
            for block_v in partition:
                if v in block_v:
                    bv = frozenset(block_v)
                    break
            nu = block_to_new_world[bu]
            nv = block_to_new_world[bv]
            new_relations[a].add((nu, nv))

    reduced = KripkeModel(
        worlds=new_worlds,
        agents=agents,
        propositions=model.propositions,
        relations=new_relations,
        valuation=new_valuation
    )
    return reduced