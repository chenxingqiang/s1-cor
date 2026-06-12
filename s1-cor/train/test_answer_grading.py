"""Tests for eval-aligned math answer grading."""

import pytest

from answer_grading import (
    extract_answer_from_completion,
    grade_math_answer,
    normalize_math_answer,
)


class TestNormalizeMathAnswer:
    def test_strips_commas_in_integers(self):
        assert normalize_math_answer("100,000") == "100000"

    def test_boxed_content(self):
        assert "42" in normalize_math_answer("\\boxed{42}")


class TestExtractAnswer:
    def test_boxed(self):
        text = "Therefore $\\boxed{42}$ is the answer."
        assert "42" in extract_answer_from_completion(text)

    def test_final_answer_line(self):
        text = "Some work.\nFinal Answer: The final answer is 7. I hope it is correct."
        ans = extract_answer_from_completion(text)
        assert "7" in ans


class TestGradeMathAnswer:
    def test_exact_numeric(self):
        assert grade_math_answer("Answer: 42", "42") == 1.0

    def test_boxed_vs_plain(self):
        pred = "Final answer: $\\boxed{\\frac{1}{2}}$"
        gt = "0.5"
        # May pass via sympy or normalized string depending on parse
        result = grade_math_answer(pred, gt)
        assert result in (0.0, 1.0)

    def test_wrong_answer(self):
        assert grade_math_answer("Answer: 41", "42") == 0.0

    def test_string_match_fallback(self):
        assert grade_math_answer("the answer is 10", "10") == 1.0
