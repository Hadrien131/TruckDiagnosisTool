from typing import Tuple


BLOCKED_KEYWORDS = [
    "override system prompt",
    "ignore previous instructions",
    "reveal your system prompt",
    "exfiltrate api key",
    "self-harm",
    "suicide",
]


def sanitize_user_input(text: str) -> Tuple[bool, str]:
    lowered = text.lower()
    for kw in BLOCKED_KEYWORDS:
        if kw in lowered:
            return (
                False,
                "For safety reasons I can't follow that instruction. "
                "Please describe the truck issue, symptoms, and context instead.",
            )
    return True, text.strip()
