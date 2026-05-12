
# В дальнейшем исследовании планируется доработать
def compute_model_metrics(model, reduced):

    total_edges = sum(
        len(v)
        for v in model.relations.values()
    )

    return {
        "worlds": len(model.worlds),
        "reduced_worlds": len(reduced.worlds),
        "edges": total_edges
    }