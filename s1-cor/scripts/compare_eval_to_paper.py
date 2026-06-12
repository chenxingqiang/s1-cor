#!/usr/bin/env python3
"""
Compare lm-evaluation-harness JSON output to README / design.md paper targets.

Usage:
    cd s1-cor
    python scripts/compare_eval_to_paper.py --results-dir results/cor-grpo-eval
    python scripts/compare_eval_to_paper.py --results path/to/file.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PAPER_TARGETS = {
    "AIME24": 56.7,
    "MATH500": 93.0,
    "GPQA": 59.6,
}

# lm_eval task name fragments → paper column
TASK_ALIASES: Dict[str, str] = {
    "aime24": "AIME24",
    "openai_math": "MATH500",
    "math500": "MATH500",
    "gpqa_diamond": "GPQA",
    "gpqa": "GPQA",
}

SCORE_KEYS = (
    "acc,none",
    "acc_norm,none",
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "acc",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare lm_eval results to paper targets")
    p.add_argument("--results", type=str, help="Single lm_eval results JSON file")
    p.add_argument(
        "--results-dir",
        type=str,
        help="Directory to scan for *.json results files",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--tolerance", type=float, default=5.0, help="± points for pass")
    return p.parse_args()


def _extract_score(task_results: Dict[str, Any]) -> Optional[float]:
    for key in SCORE_KEYS:
        if key in task_results:
            val = task_results[key]
            if isinstance(val, (int, float)):
                return float(val) * (100.0 if val <= 1.0 else 1.0)
    for key, val in task_results.items():
        if key.startswith("acc") or "exact_match" in key:
            if isinstance(val, (int, float)):
                return float(val) * (100.0 if val <= 1.0 else 1.0)
    return None


def _map_task_to_benchmark(task_name: str) -> Optional[str]:
    lower = task_name.lower()
    for fragment, bench in TASK_ALIASES.items():
        if fragment in lower:
            return bench
    return None


def scores_from_lm_eval_doc(doc: Dict[str, Any]) -> Dict[str, float]:
    """Parse one lm_eval results JSON document."""
    out: Dict[str, float] = {}
    results = doc.get("results") or {}
    for task_name, task_results in results.items():
        bench = _map_task_to_benchmark(task_name)
        if not bench or not isinstance(task_results, dict):
            continue
        score = _extract_score(task_results)
        if score is not None:
            out[bench] = score
    return out


def load_scores_from_path(path: Path) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    if path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
        merged.update(scores_from_lm_eval_doc(doc))
    elif path.is_dir():
        for fp in sorted(path.rglob("*.json")):
            try:
                doc = json.loads(fp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            merged.update(scores_from_lm_eval_doc(doc))
    return merged


def compare_scores(
    observed: Dict[str, float],
    tolerance: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bench, target in PAPER_TARGETS.items():
        actual = observed.get(bench)
        if actual is None:
            rows.append(
                {
                    "benchmark": bench,
                    "target": target,
                    "actual": None,
                    "delta": None,
                    "within_tolerance": False,
                    "status": "missing",
                }
            )
            continue
        delta = actual - target
        ok = abs(delta) <= tolerance
        rows.append(
            {
                "benchmark": bench,
                "target": target,
                "actual": round(actual, 2),
                "delta": round(delta, 2),
                "within_tolerance": ok,
                "status": "pass" if ok else "fail",
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    if not args.results and not args.results_dir:
        print("Provide --results or --results-dir", file=sys.stderr)
        return 2

    paths: List[Path] = []
    if args.results:
        paths.append(Path(args.results))
    if args.results_dir:
        paths.append(Path(args.results_dir))

    observed: Dict[str, float] = {}
    for p in paths:
        observed.update(load_scores_from_path(p))

    comparison = compare_scores(observed, args.tolerance)
    payload = {
        "paper_targets": PAPER_TARGETS,
        "observed": observed,
        "comparison": comparison,
        "tolerance_points": args.tolerance,
        "all_pass": all(r["status"] == "pass" for r in comparison),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Paper target comparison")
        for row in comparison:
            if row["status"] == "missing":
                print(f"  {row['benchmark']}: MISSING (target {row['target']})")
            else:
                mark = "✅" if row["within_tolerance"] else "❌"
                print(
                    f"  {mark} {row['benchmark']}: {row['actual']} "
                    f"(target {row['target']}, Δ{row['delta']:+.1f})"
                )

    if not observed:
        return 1
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
