# CoR development targets (meta loop layer hooks — not a full orchestrator).
# See docs/LOOPS.md and AGENTS.md.

.PHONY: loop-perceive loop-strategy loop-verify loop-product-verify loop-ablation loop-eval-smoke test

loop-perceive:
	python3 scripts/loop_perceive.py --json --skip-pytest

loop-strategy:
	python3 scripts/loop_strategy.py --json --skip-pytest

loop-verify:
	python3 scripts/loop_verify.py

loop-product-verify:
	python3 scripts/loop_product_verify.py

loop-ablation:
	python3 scripts/run_ablation_sweep.py --samples 10
	python3 scripts/run_reflection_k_ablation.py --samples 10

loop-eval-smoke:
	bash scripts/run_eval_smoke.sh

loop-r-ext-align:
	python3 scripts/run_r_ext_alignment_report.py --json --samples 10

loop-calibration:
	python3 scripts/run_calibration_report.py --json --samples 10

loop-calibration-ablation:
	python3 scripts/run_calibration_bonus_ablation.py --json --samples 10

loop-grpo-smoke:
	python3 scripts/run_grpo_reward_smoke.py --json --samples 5

loop-intrinsic-ablation:
	python3 scripts/run_intrinsic_dim_ablation.py --json --samples 10 --presets uniform,drop

loop-eval-openai-grader:
	python3 scripts/run_eval_openai_grader_report.py --json

test:
	python3 -m pytest train/ -q
