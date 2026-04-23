import json
import sys
sys.path.append('.')

from core.ast_parser import parse_formula
from kripke.builder import KripkeBuilder
from kripke.bisimulation import compute_bisimulation_quotient
from kripke.visualize import draw_kripke_model

def main():
    sample_json = '''
    {
      "id": "2f72668afd61",
      "topic": "Should voting age be lowered to 16 worldwide?",
      "text": "...",
      "agents": ["Alex", "Maya", "Jordan"],
      "propositions": [
        "Voting age should be lowered to 16 worldwide.",
        "Sixteen-year-olds are mature enough to vote responsibly.",
        "Lowering the voting age will increase political engagement among youth."
      ],
      "formulas": [
        "B_Alex(p1)",
        "B_Maya(p1)",
        "B_Maya(p2)",
        "B_Jordan(¬p2)",
        "B_Alex(B_Maya(B_Jordan(¬p2)))",
        "B_Jordan(¬p3)",
        "B_Maya(B_Alex(p3))"
      ],
      "depth": 3
    }
    '''
    data = json.loads(sample_json)
    agents = data['agents']
    propositions = data['propositions']
    formulas = data['formulas']

    builder = KripkeBuilder(agents, propositions, formulas)
    model = builder.model
    print(f"KD45 модель: {len(model.worlds)} миров")
    draw_kripke_model(model, title="Original KD45 Model", save_path="original_model.png")

    reduced = compute_bisimulation_quotient(model)
    print(f"После сокращения: {len(reduced.worlds)} миров")
    draw_kripke_model(reduced, title="Reduced KD45 Model", save_path="reduced_model.png")

    root = 0
    print("\nПроверка истинности формул в мире 0 сокращённой модели:")
    for f_str in formulas:
        ast = parse_formula(f_str)
        result = reduced.check_formula(root, ast)
        print(f"{f_str}: {result}")

if __name__ == "__main__":
    main()