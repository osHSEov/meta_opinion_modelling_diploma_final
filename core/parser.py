import json
import re
from typing import Optional, Dict


def parse_response(content: str) -> Optional[Dict]:
    content = re.sub(r'```(?:json)?\s*|\s*```', '', content).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None