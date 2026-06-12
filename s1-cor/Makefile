# CoR development targets (meta loop layer hooks — not a full orchestrator).
# See docs/LOOPS.md and AGENTS.md.

.PHONY: loop-perceive loop-verify loop-ablation loop-eval-smoke test

loop-perceive:
	python3 scripts/loop_perceive.py --json

loop-verify:
	python3 scripts/loop_verify.py

loop-ablation:
	python3 scripts/run_ablation_sweep.py --samples 10
	python3 scripts/run_reflection_k_ablation.py --samples 10

loop-eval-smoke:
	bash scripts/run_eval_smoke.sh

test:
	python3 -m pytest train/ -q
