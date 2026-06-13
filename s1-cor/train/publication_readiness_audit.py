"""CPU audit for publication-facing doc and claim alignment."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1].parent
README = REPO_ROOT / "README.md"
PUBLICATION_DOC = REPO_ROOT / "docs" / "PUBLICATION_READINESS.md"
DEFERRED_DOC = REPO_ROOT / "docs" / "DEFERRED_CLAIMS.md"
MATRIX = REPO_ROOT / "docs" / "theory_code_matrix.yaml"
FIXTURE = REPO_ROOT / "s1-cor" / "train" / "fixtures" / "lm_eval_sample_results.json"
PAPER_TEX = REPO_ROOT / "paper" / "main.tex"

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from loop_matrix import parse_matrix_components  # noqa: E402


def build_publication_readiness_report() -> Dict[str, Any]:
    issues: List[str] = []
    checks: Dict[str, bool] = {}

    checks["publication_doc_exists"] = PUBLICATION_DOC.is_file()
    checks["deferred_doc_exists"] = DEFERRED_DOC.is_file()
    checks["matrix_exists"] = MATRIX.is_file()
    checks["paper_tex_exists"] = PAPER_TEX.is_file()
    checks["readme_exists"] = README.is_file()

    readme_text = README.read_text(encoding="utf-8") if README.is_file() else ""

    checks["readme_links_publication_or_matrix"] = (
        "PUBLICATION_READINESS" in readme_text
        or "theory_code_matrix" in readme_text
        or "DEFERRED_CLAIMS" in readme_text
    )
    checks["readme_results_repro_disclaimer"] = (
        "reproduc" in readme_text.lower()
        or "EVAL_REPRODUCTION" in readme_text
        or "design.md" in readme_text
    )
    checks["readme_implementation_status"] = (
        "Implementation" in readme_text or "implementation status" in readme_text.lower()
    )

    design = (REPO_ROOT / "design.md").read_text(encoding="utf-8") if (REPO_ROOT / "design.md").is_file() else ""
    checks["design_convergence_exp_form"] = "exp(-α" in design or "exp(-α·" in design

    components = parse_matrix_components(MATRIX) if MATRIX.is_file() else []
    tiers = {c.get("id"): c.get("tier") for c in components}
    checks["benchmark_reproduction_partial"] = tiers.get("benchmark_reproduction") == "partial"
    checks["token_level_deferred"] = tiers.get("token_level_reward_chain") == "deferred"

    fixture_labeled = False
    if FIXTURE.is_file():
        raw = FIXTURE.read_text(encoding="utf-8")
        fixture_labeled = "synthetic" in raw.lower() or "_fixture_note" in raw
    checks["lm_eval_fixture_labeled"] = fixture_labeled

    for key, ok in checks.items():
        if not ok:
            issues.append(f"check failed: {key}")

    # README should not claim PDF if missing
    paper_pdf = REPO_ROOT / "paper" / "main.pdf"
    if "main.pdf" in readme_text and not paper_pdf.is_file():
        if "main.tex" not in readme_text:
            issues.append("README links main.pdf but PDF not in repo; add main.tex link")

    audit_ok = len(issues) == 0

    partial_count = sum(1 for t in tiers.values() if t == "partial")
    deferred_count = sum(1 for t in tiers.values() if t == "deferred")

    return {
        "layer": "verify",
        "report": "publication_readiness_audit",
        "checks": checks,
        "issues": issues,
        "audit_ok": audit_ok,
        "matrix_summary": {
            "partial": partial_count,
            "deferred": deferred_count,
            "benchmark_tier": tiers.get("benchmark_reproduction"),
        },
        "p0_blocker": "GPU ckpt + real lm_eval JSON → compare_eval_to_paper all pass",
        "notes": [
            "audit_ok = doc/claim hygiene on CPU; not substitute for benchmark reproduction.",
            "See docs/PUBLICATION_READINESS.md for reviewer-facing narrative.",
        ],
    }
