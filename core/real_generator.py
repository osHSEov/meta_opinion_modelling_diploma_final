import hashlib
import random
from typing import Optional, List

from .models import SyntheticSample, validate_formulas_match_text, count_nesting_depth
from .real_prompts import build_system_prompt, build_user_prompt
from .parser import parse_response


class RealStyleDatasetGenerator:
    """Generator for authentic-looking social media dialogues."""

    # Fallback topics that are known to generate good discussions
    _fallback_topics = [
        "Is it ethical to use AI-generated content for school assignments?",
        "Should social media platforms be held legally responsible for misinformation?",
        "Is remote work better for productivity than office work?",
        "Are cryptocurrency investments too risky for average investors?",
        "Should parents limit their teens' social media usage?",
        "Is cancel culture an effective tool for accountability or toxic behavior?",
        "Do video games cause aggression in players?",
        "Should plastic packaging be banned entirely?",
        "Is college still worth the cost in 2025?",
        "Should voting be made mandatory in national elections?",
        "Are electric vehicles truly better for the environment?",
        "Should airlines be allowed to overbook flights?",
        "Is the gig economy exploitation or empowerment?",
        "Should deepfake technology be illegal?",
        "Are standardized tests fair measures of student ability?",
        "Should gene editing be allowed for human enhancement?",
        "Is universal basic income a feasible solution to job loss from automation?",
        "Should religious symbols be banned in public schools?",
        "Are conspiracy theories ever justified?",
        "Should content creators be required to disclose AI use in their work?",
    ]

    def __init__(self, ollama_client, config: dict):
        self.client = ollama_client
        self.config = config

    def _get_fallback_topics(self, num_topics: int) -> List[str]:
        # Shuffle and return
        shuffled = self._fallback_topics.copy()
        random.shuffle(shuffled)
        return shuffled[:num_topics]

    def generate_topics(self, num_topics: int = 50) -> List[str]:
        """Generate diverse discussion topics."""
        prompt = build_user_prompt(
            topic="General discussion",
            max_depth=self.config["max_depth"],
            num_agents=self.config["num_agents"],
            min_props=self.config["min_props"],
            max_props=self.config["max_props"],
        )

        for attempt in range(self.config.get("max_retries", 3)):
            response = self.client.chat(
                [
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                seed_offset=attempt,
            )

            parsed = parse_response(response["message"]["content"])
            if parsed and isinstance(parsed.get("topics"), list):
                return parsed["topics"][:num_topics]

        print("Topic generation failed — using fallback pool")
        return self._get_fallback_topics(num_topics)

    def _validate_sample(self, parsed: dict) -> bool:
        required = {"text", "agents", "propositions", "formulas", "depth"}
        if not required.issubset(parsed.keys()):
            return False

        if not all([
            isinstance(parsed["agents"], list),
            isinstance(parsed["propositions"], list),
            isinstance(parsed["formulas"], list),
            isinstance(parsed["depth"], int),
            parsed["depth"] <= self.config["max_depth"],
            len(parsed["agents"]) == self.config["num_agents"],
            self.config["min_props"] <= len(parsed["propositions"]) <= self.config["max_props"],
            len(parsed["formulas"]) > 0,
        ]):
            return False

        # Validate formulas match text agents
        if not validate_formulas_match_text(parsed["formulas"], parsed["text"]):
            return False

        # Validate depth (must be exactly 2 for real-style)
        if parsed["depth"] != 2:
            return False

        return True

    def generate_sample(self, topic: str) -> Optional[SyntheticSample]:
        """Generate a single authentic-style dialogue sample."""
        prompt = build_user_prompt(
            topic,
            self.config["max_depth"],
            self.config["num_agents"],
            self.config["min_props"],
            self.config["max_props"],
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
            if parsed and self._validate_sample(parsed):
                # Generate unique ID based on content
                content_string = parsed["text"][:100] + topic
                sample_id = hashlib.md5(content_string.encode()).hexdigest()[:12]

                return SyntheticSample(
                    id=sample_id,
                    topic=topic,
                    text=parsed["text"],
                    agents=parsed["agents"],
                    propositions=parsed["propositions"],
                    formulas=parsed["formulas"],
                    depth=parsed["depth"],
                    metadata={
                        "model": self.client.model,
                        "attempt": attempt,
                        "style": "realistic",
                    },
                )

        return None
