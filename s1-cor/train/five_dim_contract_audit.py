"""CPU contract audit for five_dim_intrinsic (partial tier)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from intrinsic_weights import DIMENSION_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1].parent
FIVE_DIM_DOC = REPO_ROOT / "docs" / "FIVE_DIM_INTRINSIC.md"


def evaluate_five_dim_contract(
    correlation: Dict[str, Any],
    ablation: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    """Return per-check rows and whether the CPU contract gate passes."""
    drop_deltas = ablation.get("drop_delta_vs_uniform") or {}
    emph_deltas = ablation.get("emphasis_delta_vs_uniform") or {}
    max_drop = max((abs(v) for v in drop_deltas.values()), default=0.0)
    max_emph = max((abs(v) for v in emph_deltas.values()), default=0.0)

    rated = int(correlation.get("samples_with_self_rating") or 0)
    pearson = correlation.get("pooled_pearson_r")

    checks: Dict[str, Dict[str, Any]] = {
        "five_dimensions_defined": {
            "ok": len(DIMENSION_NAMES) == 5,
            "detail": f"dimensions={list(DIMENSION_NAMES)}",
        },
        "five_dim_doc_exists": {
            "ok": FIVE_DIM_DOC.is_file(),
            "detail": str(FIVE_DIM_DOC.relative_to(REPO_ROOT)),
        },
        "self_rating_samples": {
            "ok": rated >= 1,
            "detail": f"samples_with_self_rating={rated}",
        },
        "correlation_computed": {
            "ok": pearson is not None,
            "detail": f"pooled_pearson_r={pearson}",
        },
        "dimension_sensitivity": {
            "ok": ablation.get("most_sensitive_dimension") is not None,
            "detail": f"most_sensitive={ablation.get('most_sensitive_dimension')}",
        },
        "nonzero_ablation_delta": {
            "ok": max_drop > 1e-6 or max_emph > 1e-6,
            "detail": f"max|drop_delta|={max_drop:.4f} max|emph_delta|={max_emph:.4f}",
        },
    }
    contract_ok = all(c["ok"] for c in checks.values())
    return checks, contract_ok


def build_five_dim_contract_report(
    correlation: Dict[str, Any],
    ablation: Dict[str, Any],
) -> Dict[str, Any]:
    checks, contract_ok = evaluate_five_dim_contract(correlation, ablation)
    return {
        "layer": "verify",
        "report": "five_dim_contract",
        "matrix_id": "five_dim_intrinsic",
        "matrix_tier": "partial",
        "dimensions": list(DIMENSION_NAMES),
        "correlation_summary": {
            "pooled_pearson_r": correlation.get("pooled_pearson_r"),
            "mean_overall_calibration": correlation.get("mean_overall_calibration"),
            "most_miscalibrated_dimension": correlation.get("most_miscalibrated_dimension"),
            "samples_with_self_rating": correlation.get("samples_with_self_rating"),
        },
        "ablation_summary": {
            "most_sensitive_dimension": ablation.get("most_sensitive_dimension"),
            "emphasis_delta_vs_uniform": ablation.get("emphasis_delta_vs_uniform"),
            "drop_delta_vs_uniform": ablation.get("drop_delta_vs_uniform"),
        },
        "contract_checks": checks,
        "contract_ok": contract_ok,
        "honest_claim": (
            "Heuristic r_d + w_d ablation; not learned Q_phi. "
            "Low Pearson r is expected and must be stated in Limitations."
        ),
        "verify_commands": [
            "make loop-five-dim-contract",
            "make loop-intrinsic-correlation",
            "make loop-intrinsic-ablation",
        ],
    }
