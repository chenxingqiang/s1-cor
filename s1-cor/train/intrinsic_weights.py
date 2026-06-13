"""Parse and validate five-dimension intrinsic weights (w_d)."""

from __future__ import annotations

import json
from typing import Dict, Optional

from rewards.intrinsic import IntrinsicRewardCalculator

DIMENSION_NAMES = tuple(IntrinsicRewardCalculator.DEFAULT_WEIGHTS.keys())


def default_dimension_weights() -> Dict[str, float]:
    return IntrinsicRewardCalculator.DEFAULT_WEIGHTS.copy()


def parse_dimension_weights(raw: Optional[str]) -> Dict[str, float]:
    """Parse weights from JSON or ``dim=0.2,dim2=0.3`` format."""
    if not raw or not str(raw).strip():
        return default_dimension_weights()

    text = str(raw).strip()
    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("dimension_weights JSON must be an object")
    else:
        parsed: Dict[str, float] = {}
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"expected key=value, got: {part}")
            key, val = part.split("=", 1)
            parsed[key.strip()] = float(val.strip())

    weights = {d: float(parsed.get(d, 0.0)) for d in DIMENSION_NAMES}
    if sum(weights.values()) <= 0:
        raise ValueError("dimension weights must sum to a positive value")
    return weights


def normalize_dimension_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Return weights normalized to sum 1 (matches IntrinsicRewardCalculator)."""
    calc = IntrinsicRewardCalculator(weights)
    return dict(calc.weights)
