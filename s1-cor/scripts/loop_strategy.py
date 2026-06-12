#!/usr/bin/env python3
"""
Meta Loop Layer 2 — Strategy (策略).

Ranks theory_code_matrix gaps and eval blockers into a strategy card.
Does NOT implement changes — output feeds Layer 3 (Implement).

See docs/LOOPS.md and AGENTS.md 执行前闸门.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from loop_matrix import (  # noqa: E402
    matrix_gaps,
    parse_matrix_components,
    rank_strategy_candidates,
)


def _eval_readiness() -> Dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "scripts/check_eval_readiness.py", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ready_for_benchmark_eval": False, "blockers": ["readiness parse failed"]}


def _pytest_ok() -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "train/", "-q", "--tb=no", "-x"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def build_strategy(skip_pytest: bool = False) -> Dict[str, Any]:
    components = parse_matrix_components()
    gaps = matrix_gaps(components)
    readiness = _eval_readiness()
    cuda = readiness.get("checks", {}).get("cuda_available", False)
    pytest_pass = True if skip_pytest else _pytest_ok()

    ranked = rank_strategy_candidates(
        gaps,
        cuda_available=cuda,
        pytest_ok=pytest_pass,
    )

    focus = ranked[0] if ranked else None
    focus_id = focus.get("id") if focus else None

    strategy_questions = {
        "layer": f"matrix gap: {focus_id} ({focus.get('tier') if focus else 'none'})",
        "contract": (
            f"tier {focus.get('tier')} → verify: {focus.get('verify', 'n/a')[:80]}"
            if focus
            else "no gaps"
        ),
        "benefit": (
            "correctness / reproducibility / theory closure"
            if focus and focus.get("tier") == "partial"
            else "document proxies or GPU follow-up"
        ),
        "opportunity_cost": (
            "pytest failing — fix tests first"
            if not pytest_pass
            else "; ".join(readiness.get("blockers", [])[:2]) or "none"
        ),
    }

    return {
        "layer": "strategy",
        "meta_loop": "AGENTS.md Layer 2; pick ONE ranked gap for Layer 3",
        "pytest_ok": pytest_pass,
        "pytest_skipped": skip_pytest,
        "eval_readiness": {
            "ready": readiness.get("ready_for_benchmark_eval", False),
            "blockers": readiness.get("blockers", []),
        },
        "matrix_gaps_ranked": ranked,
        "recommended_focus": focus,
        "strategy_card": strategy_questions,
        "next_layer": "implement — minimal patch; then loop_verify + loop_product_verify",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Meta loop layer 2: strategy")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    card = build_strategy(skip_pytest=args.skip_pytest)

    if args.json:
        print(json.dumps(card, indent=2))
    else:
        print("Meta Loop — Strategy")
        focus = card.get("recommended_focus")
        if focus:
            print(f"  focus: {focus['id']} ({focus['tier']})")
            print(f"  verify: {focus.get('verify', 'n/a')}")
        print(f"  pytest: {'OK' if card['pytest_ok'] else 'FAIL'}")
        if card["eval_readiness"]["blockers"]:
            print("  eval blockers:")
            for b in card["eval_readiness"]["blockers"][:3]:
                print(f"    - {b}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
