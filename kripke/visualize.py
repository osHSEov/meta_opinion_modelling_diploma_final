import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from kripke.model import KripkeModel

def draw_kripke_model(model: KripkeModel, title: str = "Kripke Model", save_path: str = None):
    G = nx.MultiDiGraph()
    for w in model.worlds:
        true_props = sorted([f"p{i}" for i in model.valuation.get(w, set())])
        label = f"w{w}\n{','.join(true_props) if true_props else '∅'}"
        G.add_node(w, label=label)

    agent_colors = {agent: plt.cm.tab10(i) for i, agent in enumerate(model.agents)}

    for agent in model.agents:
        for (u, v) in model.relations[agent]:
            G.add_edge(u, v, agent=agent, color=agent_colors[agent])

    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=2000)

    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10)

    legend_handles = []
    for agent, color in agent_colors.items():
        edges = [(u, v) for (u, v, data) in G.edges(data=True) if data.get('agent') == agent]
        if edges:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, edge_color=color, alpha=0.7,
                                   connectionstyle='arc3,rad=0.1', arrows=True,
                                   arrowstyle='->', arrowsize=15)
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2, label=agent)
        )

    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
