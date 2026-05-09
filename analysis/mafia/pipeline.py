import argparse
from pathlib import Path

from analysis.mafia.parser import MafiaGameParser
from analysis.mafia.extractor import MafiaBeliefExtractor
from analysis.mafia.propositions import PropositionRegistry

from analysis.mafia.dialogue import DialogueBuilder
from analysis.mafia.prompts import build_meta_belief_prompt
from analysis.mafia.llm_extractor import LLMMetaBeliefExtractor

from analysis.mafia.metrics import compute_model_metrics
from analysis.mafia.serializer import save_slice_result

from kripke.builder import KripkeBuilder
from kripke.bisimulation import compute_bisimulation_quotient
from analysis.mafia.visualize_mafia import draw_mafia_kripke_model

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--game-id", required=True)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--output-dir", default="./outputs")

    parser.add_argument("--use-llm", action="store_true")

    args = parser.parse_args()

    game_parser = MafiaGameParser(
        args.logs_dir,
        args.game_id
    )

    game_parser.load()

    grouped = game_parser.grouped_introspection()

    extractor = MafiaBeliefExtractor()

    registry = PropositionRegistry()

    dialogue_builder = DialogueBuilder(
        game_parser.game_events
    )

    llm_extractor = None

    if args.use_llm:

        llm_extractor = LLMMetaBeliefExtractor({
            "backend": "ollama",
            "model": "gemma3:27b",
            "temperature": 0.2,
            "seed": 42
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    for (round_num, msg_seq), records in grouped.items():
        
        alive_agents = game_parser.alive_players(round_num)

        beliefs = extractor.extract_from_introspection(records, alive_players=alive_agents)

        formulas = []

        # introspection beliefs
        for belief in beliefs:

            prop = registry.get_or_create(
                belief.fact
            )

            agent = belief.agent.replace(" ", "_")

            formulas.append(
                f"B_{agent}({prop})"
            )

        # llm meta beliefs
        if llm_extractor:

            dialogue = dialogue_builder.build_until(
                round_num,
                msg_seq
            )

            prompt = build_meta_belief_prompt(
                dialogue
            )

            meta_beliefs = llm_extractor.extract(
                prompt
            )

            for mb in meta_beliefs:

                if mb.nested_formula:
                    formulas.append(
                        mb.nested_formula
                    )

        formulas = list(set(formulas))

        if not formulas:
            continue

        print()
        print(f"ROUND {round_num} MSG {msg_seq}")
        
        alive_agents = game_parser.alive_players(round_num)

        builder = KripkeBuilder(
            agents=alive_agents,
            propositions=registry.propositions(),
            formulas=formulas
        )

        model = builder.model

        reduced = compute_bisimulation_quotient(
            model
        )

        metrics = compute_model_metrics(
            model,
            reduced
        )

        print(metrics)

        original_path = output_dir / f"r{round_num}_m{msg_seq}.png"

        reduced_path = output_dir / f"r{round_num}_m{msg_seq}_reduced.png"

        draw_mafia_kripke_model(
            model,
            proposition_registry=registry,
            title=f"Round {round_num} Msg {msg_seq}",
            save_path=str(original_path)
        )

        draw_mafia_kripke_model(
            reduced,
            proposition_registry=registry,
            title=f"Reduced Round {round_num} Msg {msg_seq}",
            save_path=str(reduced_path)
        )

        save_slice_result(
            output_dir / f"r{round_num}_m{msg_seq}.json",
            formulas,
            metrics
        )

        print(f"Saved: {original_path}")


if __name__ == "__main__":
    main()