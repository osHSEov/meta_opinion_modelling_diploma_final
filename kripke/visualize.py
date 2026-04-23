import networkx as nx
import matplotlib.pyplot as plt

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

    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

    plt.figure(figsize=(12, 9))

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2200)

    for agent, color in agent_colors.items():
        edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['agent'] == agent]

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
            connectionstyle='arc3,rad=0.2', 
            min_source_margin=25,
            min_target_margin=25
        )

    # --- подписи ---
    nx.draw_networkx_labels(
        G,
        pos,
        nx.get_node_attributes(G, 'label'),
        font_size=10,
        font_weight='bold'
    )

    # --- легенда ---
    for agent, color in agent_colors.items():
        plt.plot([], [], color=color, label=agent, linewidth=3)

    plt.legend(title="Agents", loc='upper left', bbox_to_anchor=(1, 1))

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()