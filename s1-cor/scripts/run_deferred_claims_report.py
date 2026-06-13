#!/usr/bin/env python3
"""
CPU audit: deferred theory claims documented in matrix + DEFERRED_CLAIMS.md.

Usage:
    cd s1-cor
    python scripts/run_deferred_claims_report.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from deferred_claims_audit import build_deferred_claims_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Deferred claims CPU audit")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_deferred_claims_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Deferred claims audit")
        print(f"  deferred entries: {report['deferred_count']}")
        print(f"  audit_ok: {report['audit_ok']}")
        for entry in report["entries"]:
            mark = "✓" if entry.get("ok") else "✗"
            print(f"  {mark} {entry['id']}")
        if report["issues"]:
            print("  issues:")
            for issue in report["issues"]:
                print(f"    - {issue}")
    return 0 if report["audit_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
