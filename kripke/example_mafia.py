import sys
sys.path.append('.')

from core.ast_parser import parse_formula
from kripke.builder import KripkeBuilder
from kripke.bisimulation import compute_bisimulation_quotient
from kripke.visualize import draw_kripke_model

def main():
    data = {
        "id": "password_leak_002",
        "topic": "Кто знает пароль от сервера?",
        "text": """
Alice: Я уверена, что Bob знает пароль.
Bob: Charlie думает, что я не знаю пароль, но это неправда.
Charlie: Я слышал, как Dave говорил, что Bob знает пароль.
Dave: Я верю, что Alice ошибается насчёт Bob.
Eve: Bob считает, что Charlie ошибается. Но я знаю, что пароль не был скомпрометирован.
""",
        "agents": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
        "propositions": [
            "Bob_knows_password",      
            "Charlie_is_wrong",        
            "Dave_trusts_Alice",       
            "Password_compromised"     
        ],
        "formulas": [
            "B_Alice(p1)",
            "B_Bob(B_Charlie(~p1))",
            "B_Charlie(B_Dave(p1))",
            "B_Dave(~B_Alice(p1))",
            "B_Eve(B_Bob(p2))",
            "B_Eve(~p4)",
            "B_Charlie(B_Bob(B_Dave(~B_Alice(p1))))"
        ],
        "depth": 3
    }

    agents = data['agents']
    propositions = data['propositions']
    formulas = data['formulas']

    print("=" * 70)
    print("TOPIC")
    print("=" * 70)
    print(data["topic"])

    print("\n" + "=" * 70)
    print("DIALOGUE")
    print("=" * 70)
    print(data["text"])

    print("\n" + "=" * 70)
    print("PROPOSITIONS")
    print("=" * 70)
    for idx, prop in enumerate(propositions, 1):
        print(f"p{idx} = {prop}")

    print("\n" + "=" * 70)
    print("EPISTEMIC FORMULAS (извлечённые из диалога)")
    print("=" * 70)
    for idx, f in enumerate(formulas, 1):
        print(f"{idx:02d}. {f}")

    print("\n" + "=" * 70)
    print("BUILDING KD45 MODEL")
    print("=" * 70)

    builder = KripkeBuilder(
        agents=agents,
        propositions=propositions,
        formulas=formulas
    )
    model = builder.model
    print(f"Original KD45 model: {len(model.worlds)} worlds")
    draw_kripke_model(model, title="Original KD45 Model", save_path="original_model.png")

    reduced = compute_bisimulation_quotient(model)
    print(f"Reduced model: {len(reduced.worlds)} worlds")
    print("\nAccessibility relations:")
    for agent in reduced.agents:
        print(f"{agent}: {reduced.relations.get(agent, set())}")

    draw_kripke_model(reduced, title="Reduced KD45 Model", save_path="reduced_model.png")

    print("\n" + "=" * 70)
    print("FORMULA VALIDATION IN ROOT WORLD")
    print("=" * 70)

    root = 0
    for f_str in formulas:
        ast = parse_formula(f_str)
        result = reduced.check_formula(root, ast)
        print(f"{f_str:<50} -> {result}")

    print("\n" + "=" * 70)
    print("PROPOSITION LEGEND")
    print("=" * 70)
    for idx, prop in enumerate(propositions, 1):
        print(f"p{idx:<2} : {prop}")

if __name__ == "__main__":
    main()