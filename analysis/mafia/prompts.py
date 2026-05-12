def build_meta_belief_prompt(dialogue: str):

    return f"""
You are a world-class expert in epistemic modal logic, belief extraction,
social reasoning, and multi-agent dialogue analysis.

Your task is to extract:
1. Direct beliefs
2. Meta-beliefs (beliefs about beliefs)
3. Suspicion structures
4. Beliefs about hidden roles in Mafia

Return ONLY valid JSON.

IMPORTANT CONTEXT:
The dialogue comes from a multi-agent Mafia game.
Players discuss suspicions, trust, accusations, alliances,
and beliefs about other players' hidden roles.

--------------------------------------------------
FORMULA LANGUAGE
--------------------------------------------------

Use epistemic modal logic formulas:

- B_Alice(p1)
- B_Bob(B_Alice(p1))
- B_Carol(¬p2)

where:
- B_Agent(X) means "Agent believes X"
- ¬ means negation

--------------------------------------------------
PROPOSITIONS
--------------------------------------------------

Each proposition must be:
- atomic
- semantically clear
- written in natural English
- independent from modal operators

GOOD propositions:
- "Bailey is Mafia"
- "Jordan is trustworthy"
- "Harper suspects Bailey"
- "Logan trusts Jordan"

BAD propositions:
- "Jordan believes Bailey is Mafia"
- "Harper thinks Logan trusts Jordan"

Belief nesting belongs ONLY in formulas.

--------------------------------------------------
EXTRACTION RULES
--------------------------------------------------

Extract formulas ONLY if they are:
- explicitly stated
OR
- strongly implied by the dialogue

Examples:

"Bailey is acting suspicious."
→ B_Speaker(pX)

"I think Harper trusts Jordan."
→ B_Speaker(B_Harper(pY))

"Logan probably believes Bailey is Mafia."
→ B_Speaker(B_Logan(pX))

"I don't think Jordan is innocent."
→ B_Speaker(¬pX)

--------------------------------------------------
IMPORTANT CONSTRAINTS
--------------------------------------------------

1. Every agent in formulas MUST appear in dialogue.
2. Every proposition must be referenced by at least one formula.
3. Avoid duplicate propositions.
4. Use consistent proposition indexing:
   p1, p2, p3...
5. Maximum meta-belief depth should reflect the dialogue naturally.
6. Prefer concise propositions.
7. Preserve nested reasoning when present.

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Return ONLY valid JSON:

{{
  "agents": [
    "Jordan",
    "Bailey",
    "Harper"
  ],

  "propositions": {{
    "p1": "Bailey is Mafia",
    "p2": "Jordan is trustworthy",
    "p3": "Harper trusts Jordan"
  }},

  "formulas": [
    "B_Jordan(p1)",
    "B_Harper(B_Jordan(p1))",
    "B_Logan(B_Harper(p2))",
    "B_Bailey(¬p2)"
  ],

  "depth": 3
}}

--------------------------------------------------
DIALOGUE
--------------------------------------------------

{dialogue}

--------------------------------------------------
JSON OUTPUT
--------------------------------------------------
"""