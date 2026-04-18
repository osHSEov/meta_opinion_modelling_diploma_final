from typing import Optional


def build_system_prompt() -> str:
    return (
        "You are an expert in collecting and anonymizing real social media conversations "
        "for research purposes. Your task is to create realistic, authentic-looking dialogues "
        "that could have come from real Reddit or Twitter discussions."
    )


def build_user_prompt(
    topic: str,
    max_depth: int,
    num_agents: int,
    min_props: int,
    max_props: int,
) -> str:
    return f"""Topic: {topic}
Exactly {num_agents} users participate in this discussion.

CRITICAL REQUIREMENTS:
- This dialogue must sound like it came from Reddit (r/AskReddit, r/politics, r/technology, r/personalfinance)
- Use authentic social media language: casual, sometimes abbreviated, with reactions like "lol", "tbh", "fr", "smh"
- Users have distinct personalities: one cynical, one passionate, one measured
- Include natural interruptions, disagreements, and follow-ups
- Maximum nesting depth: {max_depth} (B-operator depth)
- Number of atomic propositions: {min_props}-{max_props}

Real Reddit dialogue characteristics to include:
- Casual language with slang/abbreviations ("I think", "I dunno", "tbh", "fr", "lmao")
- Reactionary responses ("This is so true!", "Wait, what?", "I disagree because...")
- Personal anecdotes or references to real-life situations
- Mild informal tone but respectful disagreement

Output ONLY valid JSON in this format:
{{
  "text": "User1: comment...\nUser2: reply...\nUser3: another reply...",
  "agents": ["username1", "username2", "username3"],
  "propositions": ["Proposition 1", "Proposition 2", ...],
  "formulas": ["B_username1(p1)", "B_username2(B_username1(p2))", ...],
  "depth": 2
}}

IMPORTANT: All formulas must be explicitly stated in the dialogue. Depth must be exactly 2.

Now generate the data:
Topic: {topic}
"""
