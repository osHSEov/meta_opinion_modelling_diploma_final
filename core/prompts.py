from typing import Optional


def build_system_prompt() -> str:
    # Оставляем как есть — генератору диалогов полезно знать контекст задачи
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

Generate a realistic, everyday dialogue that sounds like a natural conversation from social media,
a forum, or a group chat. Avoid academic or overly formal language. Vary the agents' perspectives,
the set of atomic propositions, and the structure of beliefs.

Return ONLY valid JSON.

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
        "You are a world-class expert at generating diverse, high-quality discussion topics "
        "that appear in everyday conversations, social media debates, and news headlines. "
        "Topics should be drawn from a wide range of areas: politics, sports, entertainment, "
        "technology, health, environment, economy, lifestyle, education, and more."
    )


def build_topic_user_prompt(num_topics: int = 20) -> str:
    return f"""Generate exactly {num_topics} diverse, realistic, and opinion-provoking topics 
that people commonly debate on social networks, forums, or in casual chats.

Cover a broad mix of domains, for example:
- Politics: elections, policies, international affairs
- Sports: team rivalries, rule changes, athlete compensation
- Technology: AI, privacy, social media impact
- Entertainment: movies, music, celebrity culture
- Lifestyle: remote work, diets, parenting
- Health: vaccines, mental health, fitness trends
- Environment: climate change, renewable energy, conservation
- Economy: inflation, cryptocurrency, gig work

Make the topics controversial or debatable but not offensive. Avoid academic jargon 
and words like "epistemic", "modal logic", or "ontology".

Output **ONLY** valid JSON in this exact format:
{{
  "topics": ["Topic one here", "Topic two here", ...]
}}

Do not add any extra text.
"""