"""Compare training R_ext extraction vs lm-eval pre-OpenAI eval path."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from answer_grading import (
    extract_answer_from_completion,
    is_math_equiv,
    last_boxed_only_string,
    normalize_math_answer,
    remove_boxed,
)

# lm-eval openai_math/utils.py pre-OpenAI extraction order
EVAL_ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*(.*)", re.DOTALL)
EVAL_SPLIT_TOKENS = ("<|im_start|>answer\n", "<|im_start|>")


def extract_lm_eval_pre_openai_answer(completion: str) -> str:
    """Mimic openai_math utils before OpenAI sampler fallback."""
    if not completion:
        return ""

    text = completion
    if EVAL_SPLIT_TOKENS[0] in text:
        text = text.split(EVAL_SPLIT_TOKENS[0])[-1]
    elif EVAL_SPLIT_TOKENS[1] in text:
        text = text.split(EVAL_SPLIT_TOKENS[1])[-1]
        if "\n" in text:
            text = "\n".join(text.split("\n")[1:])

    boxed = last_boxed_only_string(text)
    if boxed is not None:
        return remove_boxed(boxed).strip()

    matches = EVAL_ANSWER_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()

    return text.strip()


def _math_correct(prediction: str, ground_truth: str, *, use_eval_extract: bool) -> bool:
    if use_eval_extract:
        pred_raw = extract_lm_eval_pre_openai_answer(prediction)
    else:
        pred_raw = extract_answer_from_completion(prediction)
    gt_raw = extract_answer_from_completion(ground_truth)
    if not pred_raw and not gt_raw:
        return True
    if not pred_raw or not gt_raw:
        return False
    if normalize_math_answer(pred_raw) == normalize_math_answer(gt_raw):
        return True
    return is_math_equiv(pred_raw, gt_raw)


def build_path_alignment_report(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    rows: dicts with keys attempt/solution (or completion/ground_truth).
    """
    n = len(rows)
    train_correct = 0
    eval_correct = 0
    agree = 0
    openai_likely: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        attempt = row.get("attempt") or row.get("completion") or ""
        solution = row.get("solution") or row.get("ground_truth") or ""

        t_ok = _math_correct(attempt, solution, use_eval_extract=False)
        e_ok = _math_correct(attempt, solution, use_eval_extract=True)

        if t_ok:
            train_correct += 1
        if e_ok:
            eval_correct += 1
        if t_ok == e_ok:
            agree += 1
        elif t_ok and not e_ok:
            openai_likely.append(
                {
                    "index": i,
                    "reason": "train_math_ok_eval_pre_openai_fail",
                    "train_extract": extract_answer_from_completion(attempt)[:80],
                    "eval_extract": extract_lm_eval_pre_openai_answer(attempt)[:80],
                }
            )

    return {
        "samples": n,
        "train_math_accuracy": train_correct / n if n else 0.0,
        "eval_pre_openai_accuracy": eval_correct / n if n else 0.0,
        "path_agreement_rate": agree / n if n else 0.0,
        "openai_fallback_likely_count": len(openai_likely),
        "openai_fallback_sample": openai_likely[:5],
        "recommended_training_grader": "math",
    }
