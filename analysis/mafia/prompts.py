def build_meta_belief_prompt(dialogue):

    return f"""
Ты эксперт по эпистемической логике.

Извлеки:

1. beliefs
2. meta-beliefs

Формат:

{{
  "formulas": [
     "B_Alice(p1)",
     "B_Bob(B_Alice(p1))",
     "B_Carol(~p2)"
  ],

  "propositions": {{
      "p1": "Bob is Mafia",
      "p2": "Alice is Doctor"
  }}
}}

Диалог:

{dialogue}

JSON:
"""