from typing import Optional


def build_system_prompt() -> str:
    return (
        "You are a world-class expert in epistemic logic and synthetic dataset generation "
        "for benchmarks on extracting beliefs and meta-beliefs from text."
    )


def build_user_prompt(
    topic: str,
    num_agents: int,
    max_depth: int,
    min_props: int,
    max_props: int,
    style: str,
    style_description: str,
    name_pool: list,
    speech_markers: list
) -> str:
    
    markers = ", ".join(speech_markers)
    names = ", ".join(name_pool)
    
    return f"""Topic: {topic}

Exactly {num_agents} agents participate.
Maximum nesting depth of B-operators: {max_depth}.
Number of atomic propositions: from {min_props} to {max_props}.
Style: {style_description}
Speech markers typical for this style: {markers}

Generate a realistic, everyday dialogue that sounds like a natural conversation from social media,
a forum, or a group chat. Avoid academic or overly formal language. Vary the agents' perspectives,
the set of atomic propositions, and the structure of beliefs.

Return ONLY valid JSON.

STRICT RULES:
• agents — exactly {num_agents} unique names (use common names like Alex, Maya, Jordan, Sam, Taylor, Robin).
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
  IMPORTANT: Every formula must be directly expressed by someone in the dialogue.
  The agent name in B_Agent(X) must match a speaker in the text.
• depth — maximum nesting depth of any B-operator in the formulas list (integer).

CHECKLIST before final output:
1. Each formula's agent appears as a speaker in the text
2. The formula depth matches the "depth" field
3. All formulas are actually expressed in the dialogue text

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

def build_extraction_prompt(text: str) -> str:
    example1_input = """Alex: I really think that lowering the voting age to 16 would boost youth political engagement.
Alex: Honestly, I don't think sixteen‑year‑olds are mature enough to make informed decisions.
Maya: I agree with you, Alex, that lowering the age would increase engagement.
Maya: I also believe Alex believes that lowering the voting age will boost engagement.
Maya: I don't think they're mature enough either, so I believe that's false.
Jordan: From what I've read, countries that lower the voting age see higher voter turnout overall.
Jordan: I don't even believe that sixteen‑year‑olds are mature enough.
Jordan: I think Maya believes that Alex believes the engagement claim."""
    example1_output = {
        "agents": ["Alex", "Maya", "Jordan"],
        "propositions": [
            "Lowering the voting age to 16 would increase youth political engagement.",
            "Sixteen‑year‑olds are mature enough to make informed voting decisions.",
            "Countries that lower the voting age see higher voter turnout overall."
        ],
        "formulas": [
            "B_Alex(p1)",
            "B_Alex(¬p2)",
            "B_Maya(p1)",
            "B_Maya(¬p2)",
            "B_Maya(B_Alex(p1))",
            "B_Jordan(p3)",
            "¬B_Jordan(p2)",
            "B_Jordan(B_Maya(B_Alex(p1)))"
        ],
        "depth": 3
    }

    example2_input = """Mike: Honestly, I think it's totally fair that athletes get those multi‑million contracts. They generate a lot of economic revenue for the league.
Jenna: I don't think it's fair. Fans are already paying crazy ticket prices to watch games.
Luis: I believe Jenna thinks Mike believes it's fair.
Luis: And I think Mike doesn't think other professions deserve higher wages than athletes."""
    example2_output = {
        "agents": ["Mike", "Jenna", "Luis"],
        "propositions": [
            "It is fair that professional athletes receive multi‑million dollar contracts.",
            "Fans pay high ticket prices to watch games.",
            "Athletes generate significant economic revenue.",
            "Other professions deserve higher wages than athletes."
        ],
        "formulas": [
            "B_Mike(p1)",
            "B_Mike(p3)",
            "B_Jenna(p2)",
            "¬B_Jenna(p1)",
            "B_Luis(B_Jenna(B_Mike(p1)))",
            "B_Luis(¬B_Mike(p4))"
        ],
        "depth": 3
    }

    import json
    ex1_json = json.dumps(example1_output, indent=2)
    ex2_json = json.dumps(example2_output, indent=2)

    return f"""You are an expert in epistemic modal logic. Extract beliefs from the dialogue.

Output a JSON object with:
- "agents": list of unique speaker names.
- "propositions": list of the full English statements (exact wording from the dialogue, NOT "p1" placeholders).
- "formulas": list of belief formulas using B_Agent(pX) notation. Index X starts from 1 (p1 = first proposition).
- "depth": maximum nesting depth of B-operators (integer).

IMPORTANT:
- Propositions must be the original English sentences.
- Use "¬" for negation.
- Formulas refer to propositions by their 1‑based index: p1, p2, p3, ...

Examples:

Example 1:
Dialogue:
{example1_input}

Output:
{ex1_json}

Example 2:
Dialogue:
{example2_input}

Output:
{ex2_json}

Now process:
Dialogue:
{text}

JSON output:"""