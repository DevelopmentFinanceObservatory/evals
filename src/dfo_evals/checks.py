"""The convention every check evaluator follows.

Each check is one Pydantic Evals evaluator, and the evaluator's name is the check name.
A check returns either:

- ``verdict(...)``: a label whose value is a ``Verdict`` string, for checks that can
  answer more than pass or fail (``unverifiable``, ``n/a``, ``unreached``); or
- a bool: an assertion, where True is ``pass`` and False is ``fail``.

Checks never return numbers: ``run_from_report`` rejects numeric scores.
"""

from pydantic_evals.evaluators import EvaluationReason

from dfo_evals.records import Verdict


def verdict(value: Verdict, detail: str | None = None) -> EvaluationReason:
    """Build the result a check evaluator returns.

    Args:
        value: The check's verdict; becomes the label value.
        detail: Why the check reached this verdict.

    Returns:
        An evaluation reason Pydantic Evals records as a label.
    """
    return EvaluationReason(value=Verdict(value).value, reason=detail)
