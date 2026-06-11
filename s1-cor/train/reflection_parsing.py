"""
Parse multi-round reflection chains from completions and CoR dataset rows.

Supports:
  - Explicit [Round N] markers (design.md / GRPO completions)
  - thinking_trajectories list with len > 1
  - Embedded [Self-Rating: ...] checkpoints in thinking_rated (s1K-cor default)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_SELF_RATING_PATTERN = re.compile(r"\[Self-Rating:[^\]]*\]", re.IGNORECASE)
_ROUND_PATTERN = re.compile(r"\[Round (\d+)\]", re.IGNORECASE)


def extract_reflection_rounds(completion: str) -> List[str]:
    """Extract thinking chains from explicit [Round N] markers."""
    rounds = list(_ROUND_PATTERN.finditer(completion))

    if len(rounds) < 2:
        return [completion] if completion.strip() else []

    chain_sequence: List[str] = []

    for i, match in enumerate(rounds):
        start = match.end()
        end = rounds[i + 1].start() if i + 1 < len(rounds) else len(completion)
        round_content = completion[start:end].strip()

        reflection_match = re.search(r"\[Reflection\].*$", round_content, re.DOTALL)
        if reflection_match:
            round_content = round_content[: reflection_match.start()].strip()

        if round_content:
            chain_sequence.append(round_content)

    return chain_sequence if chain_sequence else [completion]


def split_thinking_by_self_ratings(thinking: str) -> List[str]:
    """Build cumulative chain snapshots at each [Self-Rating: ...] boundary.

    s1K-cor ``thinking_rated`` embeds multiple self-ratings in one draft; each
    snapshot c_k is the prefix through the k-th rating (inclusive).
    """
    if not thinking or not thinking.strip():
        return []

    matches = list(_SELF_RATING_PATTERN.finditer(thinking))
    if len(matches) < 2:
        return [thinking.strip()]

    chains: List[str] = []
    for match in matches:
        chunk = thinking[: match.end()].strip()
        if chunk:
            chains.append(chunk)

    tail = thinking[matches[-1].end() :].strip()
    if tail and chains:
        chains[-1] = f"{chains[-1]}\n{tail}".strip()

    return chains if len(chains) >= 2 else [thinking.strip()]


def extract_chain_sequence_from_text(text: str) -> List[str]:
    """Best-effort chain list from raw completion / thinking text."""
    if not text:
        return []

    explicit = extract_reflection_rounds(text)
    if len(explicit) >= 2:
        return explicit

    by_ratings = split_thinking_by_self_ratings(text)
    if len(by_ratings) >= 2:
        return by_ratings

    return [text.strip()] if text.strip() else []


def extract_chain_sequence_from_sample(sample: Dict[str, Any]) -> List[str]:
    """Derive [c_0, ..., c_K] from a CoR dataset row."""
    trajectories = sample.get("thinking_trajectories") or []
    if isinstance(trajectories, list) and len(trajectories) >= 2:
        chains = [str(t).strip() for t in trajectories if str(t).strip()]
        if len(chains) >= 2:
            return chains

    for field in ("thinking_rated", "cot", "text_cor", "text"):
        raw = sample.get(field)
        if not raw or not isinstance(raw, str):
            continue

        chains = extract_chain_sequence_from_text(raw)
        if len(chains) >= 2:
            return chains

    thinking = trajectories[0] if trajectories else ""
    if thinking:
        return [str(thinking).strip()]
    return []
