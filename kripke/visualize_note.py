from pyvis.network import Network
import networkx as nx


def draw_kripke_model_interactive(model, output_file="kripke.html"):
    G = nx.MultiDiGraph()

    for w in model.worlds:

        props = sorted([
            f"p{i}"
            for i in model.valuation.get(w, set())
        ])

        label = f"w{w}\n{','.join(props) if props else '∅'}"

        G.add_node(
            w,
            label=label,
            title=f"World w{w}<br>Props: {props}",
            color="lightblue"
        )

    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "black"
    ]

    agent_colors = {
        agent: colors[i % len(colors)]
        for i, agent in enumerate(model.agents)
    }

    for agent in model.agents:
        for (u, v) in model.relations[agent]:

            G.add_edge(
                u,
                v,
                label=agent,
                color=agent_colors[agent],
                arrows="to"
            )
    net = Network(
        height="850px",
        width="100%",
        directed=True,
        notebook=True
    )

    net.from_nx(G)

    net.force_atlas_2based()

    net.show(output_file)