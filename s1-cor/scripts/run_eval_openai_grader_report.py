#!/usr/bin/env python3
"""
CPU audit for eval-only OpenAI grading (MATH500 / GPQA).

Documents prerequisites and regex extraction smoke without calling OpenAI API.

Usage:
    cd s1-cor
    python scripts/run_eval_openai_grader_report.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from eval_openai_grader_audit import build_audit_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval OpenAI grader CPU audit")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_audit_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Eval OpenAI grader audit")
        print(f"  tasks: {', '.join(report['openai_eval_tasks']) or '(none parsed)'}")
        print(f"  OPENAI_API_KEY: {'set' if report['openai_api_key_set'] else 'unset'}")
        print(f"  regex smoke: {'OK' if report['regex_extraction_smoke']['ok'] else 'FAIL'}")
        print(f"  ready_for_openai_eval: {report['ready_for_openai_eval']}")
        if report["blockers"]:
            print("  blockers:")
            for b in report["blockers"]:
                print(f"    - {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
