import networkx as nx
import matplotlib.pyplot as plt
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
            if u != v:
                G.add_edge(u, v, agent=agent)

    loops_by_world = {
        w: sorted([agent for agent in model.agents if (w, w) in model.relations[agent]])
        for w in model.worlds
    }

    pos = nx.spring_layout(G, seed=42, k=3, iterations=80)

    plt.figure(figsize=(13, 9))

    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightblue',
        node_size=2200
    )

    for idx, (agent, color) in enumerate(agent_colors.items()):
        edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['agent'] == agent]
        if edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
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

    lines = ["Reflexive edges:"]
    for w in model.worlds:
        agents = loops_by_world[w]
        if agents:
            lines.append(f"w{w}: {', '.join(agents)}")
        else:
            lines.append(f"w{w}: ∅")

    table_text = "\n".join(lines)

    plt.gca().text(
        1.02, 0.5,
        table_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle="round,pad=0.4", facecolor="whitesmoke", alpha=0.8)
    )

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()