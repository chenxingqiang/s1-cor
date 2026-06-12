#!/usr/bin/env python3
"""
Meta Loop Layer 4 — Verify (验证).

CPU merge gate: full train/ pytest + validate_cor_logic smoke.
Does NOT run GPU benchmark eval. See docs/LOOPS.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Meta loop layer 4: verify")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    steps = [
        ([sys.executable, "-m", "pytest", "train/", "-q"], "pytest train/"),
        (
            [
                sys.executable,
                "train/validate_cor_logic.py",
                "--dataset",
                "deepseek",
                "--samples",
                str(args.samples),
            ],
            "validate_cor_logic",
        ),
    ]

    failed = []
    for cmd, name in steps:
        print(f"▶ {name}...")
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            failed.append(name)
            print(f"✗ {name} failed (exit {proc.returncode})")
        else:
            print(f"✓ {name}")

    if failed:
        print(f"\nloop_verify: FAILED ({', '.join(failed)})")
        return 1

    print("\nloop_verify: OK (meta loop layer 4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
