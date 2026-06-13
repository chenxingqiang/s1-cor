#!/usr/bin/env python3
"""
CPU audit: publication-facing docs align with matrix tiers.

Usage:
    cd s1-cor
    python scripts/run_publication_readiness_report.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from publication_readiness_audit import build_publication_readiness_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Publication readiness CPU audit")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_publication_readiness_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Publication readiness audit")
        print(f"  audit_ok: {report['audit_ok']}")
        print(f"  P0 blocker: {report['p0_blocker']}")
        for key, ok in report["checks"].items():
            print(f"  {'✓' if ok else '✗'} {key}")
        if report["issues"]:
            print("  issues:")
            for issue in report["issues"]:
                print(f"    - {issue}")
    return 0 if report["audit_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
