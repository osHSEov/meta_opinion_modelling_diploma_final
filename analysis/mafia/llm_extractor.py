import json
from services.llm_factory import build_llm_client
from .types import BeliefStatement


class LLMMetaBeliefExtractor:

    def __init__(self, config):
        self.client = build_llm_client(config)

    def extract(self, prompt):

        response = self.client.chat([
            {
                "role": "user",
                "content": prompt
            }
        ])

        text = response["message"]["content"]

        try:
            parsed = json.loads(text)
        except:
            return []

        beliefs = []

        for formula in parsed.get("formulas", []):

            beliefs.append(
                BeliefStatement(
                    agent="LLM",
                    nested_formula=formula,
                    source="llm"
                )
            )

        return beliefs