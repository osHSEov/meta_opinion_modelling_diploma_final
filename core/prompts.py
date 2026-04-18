from typing import Optional


def build_system_prompt() -> str:
    return (
        "You are a world-class expert in epistemic logic and synthetic dataset generation "
        "for benchmarks on extracting beliefs and meta-beliefs from text."
    )


def build_user_prompt(
    topic: str,
    max_depth: int,
    num_agents: int,
    min_props: int,
    max_props: int,
) -> str:
    return f"""Topic: {topic}
Exactly {num_agents} agents participate.
Maximum nesting depth of B-operators: {max_depth}.
Number of atomic propositions: from {min_props} to {max_props}.

Generate a realistic dialogue that is **different** from typical patterns. Vary the agents' perspectives, the set of atomic propositions, and the structure of beliefs. 
and return ONLY valid JSON.

STRICT RULES:
• agents — exactly {num_agents} unique names.
• propositions — list of {min_props}-{max_props} clear atomic statements (in English).
  In formulas they will be referenced as p1, p2, … (p1 = propositions[0], etc.).
• formulas — ALL beliefs and meta-beliefs actually expressed in the dialogue.
  Correct syntax examples:
  - B_Agent1(p1)
  - B_Agent2(B_Agent1(p2))
  - ¬B_Agent3(p1)
  - B_Agent1(¬B_Agent2(p3))
  Maximum nesting depth ≤ {max_depth}. Must include at least one formula with depth exactly {max_depth}.
• text — natural dialogue in the format:
  "Agent1: statement...\nAgent2: statement..."
  The dialogue must clearly express ALL formulas from the list.
• depth — maximum nesting depth of any B-operator in the formulas list (integer).

EXAMPLE (use as template only):

{{
  "text": "Anna: I am certain that p1 is true.\\nBob: I agree with Anna about p1, but she is wrong about p2.\\nCarol: Bob is right — Anna is indeed mistaken about p2.",
  "agents": ["Anna", "Bob", "Carol"],
  "propositions": ["AI will surpass human intelligence by 2030.", "Universal basic income will reduce poverty dramatically."],
  "formulas": ["B_Anna(p1)", "B_Bob(B_Anna(p1))", "B_Carol(¬B_Anna(p2))"],
  "depth": 2
}}

Now generate the data strictly according to the parameters and output **ONLY** the JSON:
{{
  "text": "...",
  "agents": [...],
  "propositions": [...],
  "formulas": [...],
  "depth": N
}}
"""


def build_topic_system_prompt() -> str:
    return (
        "You are a world-class expert at generating diverse, high-quality topics "
        "for epistemic logic and meta-opinion benchmarks."
    )


def build_topic_user_prompt(num_topics: int = 20) -> str:
    return f"""Generate exactly {num_topics} diverse, realistic, and opinion-provoking topics 
that are perfect for discussions involving beliefs, meta-beliefs, and epistemic logic in social networks.

Topics should cover technology, society, ethics, politics, science, environment, etc.
Make them controversial or debatable but not offensive.

Output **ONLY** valid JSON in this exact format:
{{
  "topics": ["Topic one here", "Topic two here", ...]
}}

Do not add any extra text.
"""