import hashlib
from typing import Optional, List

from .models import SyntheticSample, validate_formulas_match_text
from .prompts import (
    build_system_prompt,
    build_user_prompt,
    build_topic_system_prompt,
    build_topic_user_prompt,
)
from .parser import parse_response
from .styles import STYLE_DESCRIPTIONS

class MetaOpinionDatasetGenerator:
    def __init__(self, ollama_client, config: dict):
        self.client = ollama_client
        self.config = config
        self._fallback_topics = [
            "The ethical implications of AI surveillance in smart cities",
            "Universal basic income as a solution to automation-driven unemployment",
            "The role of social media in shaping public opinion on climate change",
            "Genetic editing in humans: benefits versus moral risks",
            "Cryptocurrency regulation and its impact on global finance",
            "Remote work policies and their effect on company productivity and culture",
            "The future of space colonization and international cooperation",
            "Vaccine mandates versus individual freedom during health crises",
            "The influence of cancel culture on free speech and public discourse",
            "Quantum computing threats to current encryption standards",
            "The balance between national security and personal privacy in the digital age",
            "Artificial general intelligence timelines and existential risks",
            "The feasibility and costs of a full renewable energy transition by 2035",
            "The metaverse and its potential to reshape human social interactions",
            "Immigration policies and cultural integration challenges in Europe",
        ]

    def _get_fallback_topics(self, num_topics: int) -> List[str]:
        return self._fallback_topics[:num_topics]

    def generate_topics(self, num_topics: int = 20) -> List[str]:
        
        prompt = build_topic_user_prompt(num_topics)
        
        for attempt in range(self.config.get("max_retries", 3)):
            response = self.client.chat(
                [
                    {"role": "system", "content": build_topic_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                seed_offset=attempt,
            )

            parsed = parse_response(response["message"]["content"])
            if (
                parsed
                and isinstance(parsed.get("topics"), list)
                and len(parsed["topics"]) >= 5  
            ):
                return parsed["topics"][:num_topics]

        print("Topic generation failed — using fallback pool")
        return self._get_fallback_topics(num_topics)

    def _validate_sample(
        self, 
        parsed: dict,
        expected_num_agents: int,
        min_props: int,
        max_props: int,
        max_depth: int,
        ) -> bool:
        required = {"text", "agents", "propositions", "formulas", "depth"}
        
        if not required.issubset(parsed.keys()):
            return False

        if (
            not isinstance(parsed["agents"], list)
            or not isinstance(parsed["propositions"], list)
            or not isinstance(parsed["formulas"], list)
            or not isinstance(parsed["depth"], int)
        ):
            return False

        if len(parsed["agents"]) != expected_num_agents:
            return False

        if not (
            min_props <= len(parsed["propositions"]) <= max_props
        ):
            return False

        if len(parsed["formulas"]) == 0:
            return False
        
        if parsed["depth"] > max_depth:
            return False

        if not validate_formulas_match_text(
            parsed["formulas"],
            parsed["text"]
        ):
            return False

        return True

        

    def generate_sample(
        self, 
        topic: str,
        num_agents: int,
        max_depth: int,
        min_props: int,
        max_props: int,
        style: str,
        ) -> Optional[SyntheticSample]:
        
        style_info = STYLE_DESCRIPTIONS[style]

        prompt = build_user_prompt(
            topic,
            num_agents=num_agents,
            max_depth=max_depth,
            min_props=min_props,
            max_props=max_props,
            style=style,
            style_description=style_info["description"],
            name_pool=style_info["name_pool"],
            speech_markers=style_info["speech_markers"],
        )

        for attempt in range(self.config["max_retries"]):
            response = self.client.chat(
                [
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                seed_offset=attempt,
            )

            parsed = parse_response(response["message"]["content"])
            if parsed and self._validate_sample(parsed, num_agents, min_props, max_props, max_depth):
                sample_id = hashlib.md5(
                    (parsed["text"] + topic).encode()
                ).hexdigest()[:12]

                return SyntheticSample(
                    id=sample_id,
                    topic=topic,
                    text=parsed["text"],
                    agents=parsed["agents"],
                    propositions=parsed["propositions"],
                    formulas=parsed["formulas"],
                    depth=parsed["depth"],
                    metadata={"model": self.client.model, "attempt": attempt},
                )

        return None