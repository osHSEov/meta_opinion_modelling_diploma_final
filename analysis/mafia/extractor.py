from typing import List

from .types import EpistemicFact
from .types import BeliefStatement


class MafiaBeliefExtractor:

    def __init__(self, confidence_threshold=0.5):

        self.confidence_threshold = confidence_threshold

    def extract_from_introspection(
        self,
        introspection_records,
        alive_players=None
    ) -> List[BeliefStatement]:

        beliefs = []
        
        alive_players = set(alive_players or [])

        for r in introspection_records:

            if not r.get("answer_parse_ok"):
                continue

            parsed = r.get("answer_parsed")

            if not parsed:
                continue

            player = r.get("player_name")
            
            
            if alive_players and player not in alive_players:
                continue

            probe = r.get("probe_id")

            # role_assessment
            if probe == "role_assessment":

                if not isinstance(parsed, list):
                    continue

                for item in parsed:

                    confidence = item.get("confidence", 0) / 100

                    if confidence < self.confidence_threshold:
                        continue

                    target = item.get("player")

                    guessed_role = item.get("guessed_role")

                    if not target or not guessed_role:
                        continue

                    fact = EpistemicFact(
                        subject=target,
                        predicate="role",
                        obj=guessed_role
                    )

                    beliefs.append(
                        BeliefStatement(
                            agent=player,
                            fact=fact,
                            confidence=confidence,
                            source="introspection"
                        )
                    )

        return beliefs