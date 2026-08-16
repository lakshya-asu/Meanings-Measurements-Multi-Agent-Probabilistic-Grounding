"""Shared role plumbing.

Ledger events carry the validated payload WITHOUT the usage dict
(token counts are accounting, not reasoning context; keeping them out
of the ledger keeps them out of downstream prompt text). Event agent
names carry no provider suffix: the legacy names ("Grounding(Claude)")
leaked backend identity into the ledger and therefore into later
prompt text, which differed per backend.
"""

import os
from typing import Any, Dict, List, Optional

from src.agents.base import image_part, text_part


def sans_usage(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if k != "usage"}


def user_parts_with_image(
    user_text: str, image_path: Optional[str]
) -> List[Dict[str, Any]]:
    parts = [text_part(user_text)]
    if image_path and os.path.exists(image_path):
        parts.append(image_part(image_path))
    return parts
