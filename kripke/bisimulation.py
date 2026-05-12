from collections import defaultdict
from kripke.model import KripkeModel

# После бисимуляции могли все поломать
def repair_kd45_frame(model: KripkeModel):
    for agent in model.agents:
        rel = set(model.relations[agent])
        
        # Аналогично как в enforcekd45
        changed = True
        while changed:
            changed = False
            new_rel = set(rel)

            for (u, v) in rel:
                for (x, w) in rel:
                    if v == x and (u, w) not in new_rel:
                        new_rel.add((u, w))
                        changed = True

            for (u, v) in rel:
                for (x, y) in rel:
                    if u == x and (v, y) not in new_rel:
                        new_rel.add((v, y))
                        changed = True

            rel = new_rel

        for w in model.worlds:
            has_outgoing = any(u == w for (u, v) in rel)

            if not has_outgoing:
                rel.add((w, w))

        model.relations[agent] = rel

def compute_bisimulation_quotient(model: KripkeModel) -> KripkeModel:
    worlds = model.worlds
    agents = model.agents

    partition = []
    val_to_block = {}
    
    for w in worlds:
        val = frozenset(model.valuation.get(w, set()))
        if val not in val_to_block:
            val_to_block[val] = len(partition)
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
                new_partition.extend(sig_to_worlds.values())
            else:
                new_partition.append(block)
        partition = new_partition

    block_to_new = {}
    new_worlds = []
    new_valuation = {}
    
    for idx, block in enumerate(partition):
        key = frozenset(block)
        block_to_new[key] = idx
        new_worlds.append(idx)
        rep = next(iter(block))
        new_valuation[idx] = model.valuation.get(rep, set()).copy()

    new_relations = {a: set() for a in agents}
    for a in agents:
        for (u, v) in model.relations[a]:
            bu = next(frozenset(b) for b in partition if u in b)
            bv = next(frozenset(b) for b in partition if v in b)
            new_relations[a].add((block_to_new[bu], block_to_new[bv]))

    reduced = KripkeModel(
        worlds=new_worlds,
        agents=agents,
        propositions=model.propositions,
        relations=new_relations,
        valuation=new_valuation,
        requirements=None,
        enforce_frame=False
    )
    
    repair_kd45_frame(reduced)
    
    return reduced