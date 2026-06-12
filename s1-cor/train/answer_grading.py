"""
Math answer extraction and grading aligned with lm-eval metamathqa / openai_math.

Training R_ext defaults to string match; this module provides the eval-aligned path
(boxed extraction, normalize_final_answer, sympy equivalence) without OpenAI API.
"""

from __future__ import annotations

import re
import signal
from typing import Callable, Optional

ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


class _Timeout:
    def __init__(self, seconds: int = 5, error_message: str = "Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        return s[len(left) :]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return s


def normalize_math_answer(final_answer: str) -> str:
    """Normalize answer (Lewkowycz et al. appendix D via lm-eval metamathqa)."""
    if final_answer is None:
        return ""

    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def extract_answer_from_completion(text: str) -> str:
    """Extract final answer from model completion (boxed / Answer: / final answer)."""
    if not text:
        return ""

    patterns = [
        r"(?i)final answer:\s*(.+?)(?:\n|$)",
        r"(?i)the final answer is\s*(.+?)(?:\.|I hope|$)",
        r"(?i)answer:\s*(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            return m.group(1).strip()

    m = ANSWER_PATTERN.search(text)
    if m:
        return m.group(1).strip()

    boxed = last_boxed_only_string(text)
    if boxed:
        try:
            return remove_boxed(boxed).strip()
        except Exception:
            pass

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    candidate = lines[-1] if lines else text.strip()
    return _strip_answer_prefix(candidate)


_ANSWER_PREFIXES = (
    "the answer is",
    "answer:",
    "final answer:",
    "therefore,",
    "thus,",
    "so,",
)


def _strip_answer_prefix(text: str) -> str:
    t = text.strip()
    low = t.lower()
    for prefix in _ANSWER_PREFIXES:
        if low.startswith(prefix):
            return t[len(prefix) :].strip()
    return t


def is_math_equiv(x1: str, x2: str) -> bool:
    """Sympy LaTeX equivalence (lm-eval metamathqa is_equiv)."""
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
    except ImportError:
        return normalize_math_answer(x1) == normalize_math_answer(x2)

    try:
        with _Timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                return False

            try:
                return sympy.simplify(diff) == 0
            except ValueError:
                return False
    except TimeoutError:
        return False
    except Exception:
        return False


def grade_math_answer(prediction: str, ground_truth: str) -> float:
    """
    Grade prediction vs ground truth using eval-aligned normalization.

    Returns 1.0 if equivalent, else 0.0. OpenAI judge path in lm-eval is not used.
    """
    pred_raw = extract_answer_from_completion(prediction)
    gt_raw = extract_answer_from_completion(ground_truth)

    pred = normalize_math_answer(pred_raw)
    gt = normalize_math_answer(gt_raw)

    if not pred or not gt:
        return 0.0

    if pred == gt:
        return 1.0

    replace_with_nothing = ["\\", " ", "right", "left", "le"]
    pred_rep, gt_rep = pred, gt
    for r in replace_with_nothing:
        pred_rep = pred_rep.replace(r, "")
        gt_rep = gt_rep.replace(r, "")
    if pred_rep == gt_rep:
        return 1.0

    if is_math_equiv(pred, gt):
        return 1.0

    return 0.0


def make_math_grader_fn() -> Callable[[str, str], float]:
    """Factory for RewardCalculator.calculate_external_reward(grader_fn=...)."""
    return lambda pred, gt: grade_math_answer(pred, gt)
