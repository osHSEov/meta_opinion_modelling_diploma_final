class DialogueBuilder:

    def __init__(self, events):
        self.events = events

    def build_until(self, round_num, msg_seq):

        lines = []

        for e in self.events:

            if e.get("round", 0) > round_num:
                break

            if e.get("kind") in [
                "day_discuss",
                "night_mafia",
                "intro"
            ]:

                player = e.get("player")
                response = e.get("response")

                if response:
                    lines.append(f"{player}: {response}")

            if (
                e.get("round", 0) == round_num
                and e.get("public_msg_seq", -1) > msg_seq
            ):
                break

        return "\n".join(lines)