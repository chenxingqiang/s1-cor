"""Parse docs/theory_code_matrix.yaml for meta/product loop tooling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = REPO_ROOT / "docs" / "theory_code_matrix.yaml"

TIER_ORDER = {
    "partial": 0,
    "heuristic": 1,
    "deferred": 2,
    "implemented": 99,
    "unsupported": 100,
}

GPU_BLOCKED_IDS = frozenset({"benchmark_reproduction"})


def parse_matrix_components(matrix_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = matrix_path or DEFAULT_MATRIX
    if not path.is_file():
        return []

    components: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- id:"):
            if current.get("id"):
                components.append(current)
            current = {"id": stripped.split("id:", 1)[1].strip()}
        elif current and stripped.startswith("tier:"):
            current["tier"] = stripped.split("tier:", 1)[1].strip()
        elif current and stripped.startswith("verify:"):
            raw = stripped.split("verify:", 1)[1].strip()
            current["verify"] = raw.strip('"')
        elif current and stripped.startswith("notes:"):
            raw = stripped.split("notes:", 1)[1].strip()
            current["notes"] = raw.strip('"')
        elif current and stripped.startswith("code:"):
            raw = stripped.split("code:", 1)[1].strip()
            current["code"] = raw.strip('"')

    if current.get("id"):
        components.append(current)

    return components


def matrix_tier_counts(components: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for comp in components:
        tier = comp.get("tier", "unknown")
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def matrix_gaps(components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        comp
        for comp in components
        if comp.get("tier") in ("partial", "heuristic", "deferred")
    ]


def _cpu_actionable(comp: Dict[str, Any], cuda_available: bool) -> bool:
    tier = comp.get("tier")
    cid = comp.get("id", "")
    if tier == "deferred" and not comp.get("code"):
        return False
    if cid in GPU_BLOCKED_IDS and not cuda_available:
        return False
    return True


def rank_strategy_candidates(
    gaps: List[Dict[str, Any]],
    *,
    cuda_available: bool = False,
    pytest_ok: bool = True,
) -> List[Dict[str, Any]]:
    """Rank backlog items for Layer 2 strategy (lower score = higher priority)."""

    def score(comp: Dict[str, Any]) -> float:
        tier = comp.get("tier", "deferred")
        base = float(TIER_ORDER.get(tier, 50))

        cid = comp.get("id", "")
        if cid in GPU_BLOCKED_IDS and not cuda_available:
            base += 40.0

        if tier == "deferred" and not comp.get("code"):
            base += 15.0

        if not pytest_ok:
            base += 100.0

        return base

    ranked = sorted(gaps, key=score)
    out: List[Dict[str, Any]] = []
    for i, comp in enumerate(ranked):
        row = dict(comp)
        row["priority_rank"] = i + 1
        row["cpu_actionable"] = _cpu_actionable(comp, cuda_available)
        out.append(row)
    return out
