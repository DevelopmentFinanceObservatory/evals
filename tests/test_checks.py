"""Tests for the check evaluator convention."""

import pytest
from pydantic_evals.evaluators import EvaluationReason

from dfo_evals.checks import verdict
from dfo_evals.records import Verdict


def test_verdict_label_value_is_the_verdict_string():
    result = verdict(Verdict.NOT_APPLICABLE, "no unit asked")

    assert result == EvaluationReason(value="n/a", reason="no unit asked")
    assert type(result.value) is str


def test_verdict_detail_defaults_to_none():
    assert verdict(Verdict.PASS).reason is None


def test_verdict_rejects_a_string_that_is_not_a_verdict():
    with pytest.raises(ValueError, match="maybe"):
        verdict("maybe")  # ty: ignore[invalid-argument-type]  # exercising the runtime guard
