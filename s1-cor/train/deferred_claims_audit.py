"""CPU audit: deferred matrix entries are documented honestly."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1].parent
MATRIX = REPO_ROOT / "docs" / "theory_code_matrix.yaml"
DEFERRED_DOC = REPO_ROOT / "docs" / "DEFERRED_CLAIMS.md"

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from loop_matrix import parse_matrix_components  # noqa: E402

REQUIRED_DEFERRED_IDS = ("token_level_reward_chain", "dual_coupling_phi")


def _doc_text() -> str:
    if not DEFERRED_DOC.is_file():
        return ""
    return DEFERRED_DOC.read_text(encoding="utf-8")


def build_deferred_claims_report() -> Dict[str, Any]:
    components = parse_matrix_components(MATRIX)
    deferred = [c for c in components if c.get("tier") == "deferred"]
    doc = _doc_text()

    entries: List[Dict[str, Any]] = []
    issues: List[str] = []

    for comp in deferred:
        cid = comp.get("id", "")
        notes = comp.get("notes", "")
        code = comp.get("code")
        row: Dict[str, Any] = {
            "id": cid,
            "tier": comp.get("tier"),
            "code": code,
            "notes": notes,
            "documented_in_deferred_claims": cid in doc if doc else False,
        }

        if not notes:
            issues.append(f"{cid}: missing matrix notes")
            row["ok"] = False
        elif not row["documented_in_deferred_claims"]:
            issues.append(f"{cid}: not referenced in DEFERRED_CLAIMS.md")
            row["ok"] = False
        else:
            row["ok"] = True

        if cid == "token_level_reward_chain":
            if code not in (None, "null", ""):
                issues.append(f"{cid}: expected code null, got {code}")
                row["ok"] = False
            row["substitute"] = "chain-level R_int + reflection rewards"

        if cid == "dual_coupling_phi":
            row["substitute"] = "calibration_proxy_phi (ECE proxy, θ-only GRPO)"

        entries.append(row)

    for required_id in REQUIRED_DEFERRED_IDS:
        if not any(e["id"] == required_id for e in entries):
            issues.append(f"missing deferred matrix entry: {required_id}")

    if not doc:
        issues.append("DEFERRED_CLAIMS.md missing")

    audit_ok = len(issues) == 0 and len(deferred) >= len(REQUIRED_DEFERRED_IDS)

    return {
        "layer": "verify",
        "report": "deferred_claims_audit",
        "deferred_count": len(deferred),
        "entries": entries,
        "issues": issues,
        "doc_path": str(DEFERRED_DOC.relative_to(REPO_ROOT)),
        "audit_ok": audit_ok,
        "notes": [
            "Deferred ≠ bug; honest downgrade for paper/engineering narrative.",
            "Do not promote to implemented without code + pytest + ablation.",
        ],
    }
