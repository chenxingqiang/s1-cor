#!/usr/bin/env bash
# CPU smoke: lm_eval dummy model, 2 samples. Not a paper benchmark.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HARNESS="${ROOT}/eval/lm-evaluation-harness"
OUT="${ROOT}/results/eval_smoke_dummy"

if [[ ! -d "${HARNESS}" ]]; then
  echo "missing lm-evaluation-harness at ${HARNESS}" >&2
  exit 1
fi

if ! python3 -c "import lm_eval" 2>/dev/null; then
  echo "Installing lm_eval (CPU, no vllm extra)..."
  pip install -q -e "${HARNESS}"
fi

mkdir -p "${OUT}"
cd "${HARNESS}"
echo "Running lm_eval dummy smoke (limit 2)..."
lm_eval --model dummy \
  --tasks aime24_nofigures \
  --limit 2 \
  --batch_size auto \
  --output_path "${OUT}"

echo "Smoke OK. Results under ${OUT}"
python3 "${ROOT}/scripts/compare_eval_to_paper.py" --results-dir "${OUT}" || true
