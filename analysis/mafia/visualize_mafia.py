import networkx as nx
import matplotlib.pyplot as plt
import textwrap


def _fact_to_text(fact):

    if fact is None:
        return "unknown"

    if fact.predicate == "role":

        return f"{fact.subject} is {fact.obj}"

    if fact.obj:
        return f"{fact.subject} {fact.predicate} {fact.obj}"

    return f"{fact.subject} {fact.predicate}"


def draw_mafia_kripke_model(
    model,
    proposition_registry,
    title="Mafia Kripke Model",
    save_path=None
):

    G = nx.MultiDiGraph()

    for w in model.worlds:

        props = sorted([
            f"p{p}" if isinstance(p, int) else str(p)
            for p in model.valuation.get(w, set())
        ])

        label = f"w{w}"

        if props:
            label += "\n" + ",".join(props)

        else:
            label += "\n∅"

        G.add_node(w, label=label)


    agent_colors = {
        agent: plt.cm.tab10(i)
        for i, agent in enumerate(model.agents)
    }

    for agent in model.agents:

        for (u, v) in model.relations[agent]:

            if u != v:

                G.add_edge(
                    u,
                    v,
                    agent=agent
                )

    loops_by_world = {
        w: sorted([
            agent
            for agent in model.agents
            if (w, w) in model.relations[agent]
        ])
        for w in model.worlds
    }

    pos = nx.spring_layout(
        G,
        seed=42,
        k=3,
        iterations=80
    )

    fig = plt.figure(figsize=(18, 11))

    ax = fig.add_axes([0.05, 0.08, 0.62, 0.84])

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color='lightblue',
        node_size=2600,
        ax=ax
    )

    for idx, (agent, color) in enumerate(agent_colors.items()):

        edges = [
            (u, v)
            for (u, v, d) in G.edges(data=True)
            if d['agent'] == agent
        ]

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
                arrowsize=24,
                connectionstyle=f'arc3,rad={0.15 + idx * 0.1}',
                min_source_margin=25,
                min_target_margin=25,
                ax=ax
            )

    nx.draw_networkx_labels(
        G,
        pos,
        nx.get_node_attributes(G, 'label'),
        font_size=9,
        font_weight='bold',
        ax=ax
    )

    for agent, color in agent_colors.items():

        ax.plot(
            [],
            [],
            color=color,
            label=agent,
            linewidth=3
        )

    ax.legend(
        title="Agents",
        loc='upper left',
        bbox_to_anchor=(1.02, 1)
    )

    loop_lines = ["Reflexive edges:\n"]

    for w in model.worlds:

        agents = loops_by_world[w]

        if agents:
            loop_lines.append(
                f"w{w}: {', '.join(agents)}"
            )
        else:
            loop_lines.append(f"w{w}: ∅")

    loop_text = "\n".join(loop_lines)

    ax.text(
        1.02,
        0.55,
        loop_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="whitesmoke",
            alpha=0.9
        )
    )

    prop_lines = ["Propositions:\n"]

    for prop in sorted(proposition_registry.propositions()):

        fact = proposition_registry.decode(prop)

        fact_text = _fact_to_text(fact)

        wrapped = textwrap.fill(
            fact_text,
            width=35
        )

        prop_lines.append(
            f"{prop}: {wrapped}"
        )

    prop_text = "\n\n".join(prop_lines)

    ax.text(
        -0.42,
        1.0,
        prop_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="lightyellow",
            alpha=0.95
        )
    )

    ax.set_title(title, fontsize=15)

    ax.axis('off')

    if save_path:

        plt.savefig(
            save_path,
            dpi=170,
            bbox_inches='tight'
        )

    plt.show()