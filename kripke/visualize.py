import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

def draw_kripke_model(model, title="Kripke Model", save_path=None):
    G = nx.MultiDiGraph()

    for w in model.worlds:
        props = sorted([f"p{i}" for i in model.valuation.get(w, set())])
        label = f"w{w}\n{','.join(props) if props else '∅'}"
        G.add_node(w, label=label)

    agent_colors = {agent: plt.cm.tab10(i) for i, agent in enumerate(model.agents)}

    for agent in model.agents:
        for (u, v) in model.relations[agent]:
            G.add_edge(u, v, agent=agent)

    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=50)

    plt.figure(figsize=(12, 9))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2200)

    for idx, (agent, color) in enumerate(agent_colors.items()):
        # Обычные рёбра (между разными мирами)
        non_loop_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['agent'] == agent and u != v]
        if non_loop_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=non_loop_edges,
                edge_color=[color],
                width=2.5,
                alpha=0.9,
                arrows=True,
                arrowstyle='-|>',
                arrowsize=25,
                connectionstyle=f'arc3,rad={0.15 + idx * 0.12}',
                min_source_margin=25,
                min_target_margin=25
            )

        # Петли (self-loops)
        loop_edges = [
        (u, v)
        for (u, v, d) in G.edges(data=True)
        if d['agent'] == agent and u == v
            ]

        for loop_idx, (u, v) in enumerate(loop_edges):

            x, y = pos[u]

            # Смещение центра петли
            angle = 2 * np.pi * (idx + loop_idx) / max(1, len(agent_colors))
            dx = 0.08 * np.cos(angle)
            dy = 0.08 * np.sin(angle)

            loop = FancyArrowPatch(
                (x + dx, y + dy),
                (x + dx + 0.001, y + dy + 0.001),
                connectionstyle=f"arc3,rad={0.5 + idx * 0.15}",
                arrowstyle='-|>',
                mutation_scale=20,
                color=color,
                linewidth=2.5,
                alpha=0.9
            )

            plt.gca().add_patch(loop)

    # Подписи узлов
    nx.draw_networkx_labels(
        G,
        pos,
        nx.get_node_attributes(G, 'label'),
        font_size=10,
        font_weight='bold'
    )


    for agent, color in agent_colors.items():
        plt.plot([], [], color=color, label=agent, linewidth=3)

    plt.legend(title="Agents", loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()