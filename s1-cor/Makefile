# CoR development targets (meta loop layer hooks — not a full orchestrator).
# See docs/LOOPS.md and AGENTS.md.

.PHONY: loop-perceive loop-strategy loop-verify loop-product-verify loop-ablation loop-eval-smoke loop-05b-theory loop-five-dim-contract test

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

loop-intrinsic-scale:
	python3 scripts/run_intrinsic_scale_report.py --json --samples 10

loop-intrinsic-correlation:
	python3 scripts/run_self_rating_intrinsic_correlation_report.py --json --samples 10

loop-deferred-claims:
	python3 scripts/run_deferred_claims_report.py --json

loop-publication-ready:
	python3 scripts/run_publication_readiness_report.py --json

loop-eval-openai-grader:
	python3 scripts/run_eval_openai_grader_report.py --json

loop-eval-grading-path:
	python3 scripts/run_eval_grading_path_report.py --json --samples 10

loop-benchmark-repro:
	python3 scripts/run_benchmark_reproduction_report.py --json

loop-05b-theory:
	python3 scripts/run_05b_theory_verify.py --json --strict --samples 10

loop-five-dim-contract:
	python3 scripts/run_five_dim_contract_report.py --json --strict --samples 10

test:
	python3 -m pytest train/ -q
