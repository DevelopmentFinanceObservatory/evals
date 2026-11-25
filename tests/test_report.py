"""Tests for converting a Pydantic Evals report into a run."""

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

from dfo_evals.checks import verdict
from dfo_evals.page import build_page
from dfo_evals.records import FaultClass, RunManifest, Verdict, read_run, write_run
from dfo_evals.report import TASK_CHECK_NAME, input_digest, run_from_report

STARTED_AT = datetime(2026, 11, 24, 9, 0, tzinfo=UTC)
MANIFEST = RunManifest(
    suite="toy",
    run_id="run-1",
    started_at=STARTED_AT,
    finished_at=STARTED_AT + timedelta(minutes=1),
    revision="abc123",
    environment={},
    dataset_digest="sha256:dataset",
)


async def double(inputs: dict[str, int]) -> int:
    if inputs["x"] < 0:
        raise RuntimeError("negative input")
    return inputs["x"] * 2


def country_match(ctx: EvaluatorContext[Any, int, Any]) -> EvaluationReason:
    return verdict(Verdict.UNVERIFIABLE, "source unreachable")


def unit_match(ctx: EvaluatorContext[Any, int, Any]) -> EvaluationReason:
    return verdict(Verdict.NOT_APPLICABLE)


def is_even(ctx: EvaluatorContext[Any, int, Any]) -> EvaluationReason:
    return EvaluationReason(value=ctx.output % 2 == 0, reason=f"output {ctx.output}")


def is_big(ctx: EvaluatorContext[Any, int, Any]) -> bool:
    return ctx.output > 10


def free_text(ctx: EvaluatorContext[Any, int, Any]) -> str:
    return "looks fine"


def closeness(ctx: EvaluatorContext[Any, int, Any]) -> float:
    return 0.5


def answer(ctx: EvaluatorContext[Any, int, Any]) -> bool:
    return ctx.output > 0


def crashes(ctx: EvaluatorContext[Any, int, Any]) -> bool:
    raise KeyError("missing reference")


def evaluate(
    cases: list[Case[dict[str, int], int, Any]], *evaluators: Any, repeat: int = 1
) -> EvaluationReport[Any, Any, Any]:
    dataset = Dataset[dict[str, int], int, Any](
        name="toy", cases=cases, evaluators=list(evaluators)
    )
    return dataset.evaluate_sync(double, progress=False, repeat=repeat)


def by_name(checks: tuple[Any, ...]) -> dict[str, Any]:
    return {check.name: check for check in checks}


def test_label_verdicts_including_unverifiable_and_not_applicable_land_in_checks():
    report = evaluate([Case(name="a", inputs={"x": 1})], country_match, unit_match)

    (result,) = run_from_report(report=report, manifest=MANIFEST).results
    checks = by_name(result.checks)

    assert checks["country_match"].verdict is Verdict.UNVERIFIABLE
    assert checks["country_match"].detail == "source unreachable"
    assert checks["unit_match"].verdict is Verdict.NOT_APPLICABLE
    assert result.fault is None


def test_bool_assertions_map_to_pass_and_fail_with_their_reason():
    report = evaluate([Case(name="a", inputs={"x": 3})], is_even, is_big)

    (result,) = run_from_report(report=report, manifest=MANIFEST).results
    checks = by_name(result.checks)

    assert checks["is_even"].verdict is Verdict.PASS
    assert checks["is_even"].detail == "output 6"
    assert checks["is_big"].verdict is Verdict.FAIL
    assert checks["is_big"].detail is None


def test_a_label_that_is_not_a_verdict_raises_naming_case_and_evaluator():
    report = evaluate([Case(name="case-a", inputs={"x": 1})], free_text)

    with pytest.raises(ValueError, match=r"case 'case-a'.*evaluator 'free_text'"):
        run_from_report(report=report, manifest=MANIFEST)


def task(ctx: EvaluatorContext[Any, int, Any]) -> bool:
    return False


def test_an_evaluator_named_task_raises_because_the_name_is_reserved():
    report = evaluate([Case(name="case-a", inputs={"x": 1})], task, is_even)

    with pytest.raises(ValueError, match=r"case 'case-a'.*'task' is reserved"):
        run_from_report(report=report, manifest=MANIFEST)


def test_a_numeric_score_raises():
    report = evaluate([Case(name="case-a", inputs={"x": 1})], closeness)

    with pytest.raises(ValueError, match=r"case 'case-a'.*closeness.*numeric"):
        run_from_report(report=report, manifest=MANIFEST)


def test_an_evaluator_that_raises_becomes_unverifiable_with_a_grader_fault():
    report = evaluate([Case(name="a", inputs={"x": 1})], crashes, is_even)

    (result,) = run_from_report(report=report, manifest=MANIFEST).results
    checks = by_name(result.checks)

    assert checks["crashes"].verdict is Verdict.UNVERIFIABLE
    assert checks["crashes"].detail is not None
    assert "missing reference" in checks["crashes"].detail
    assert checks["is_even"].verdict is Verdict.PASS
    assert result.fault is FaultClass.GRADER


def test_a_crash_in_another_check_leaves_a_trial_scored_by_its_headline():
    report = evaluate([Case(name="a", inputs={"x": 1})], crashes, is_even)
    manifest = MANIFEST.model_copy(update={"headline_check": "is_even"})

    (result,) = run_from_report(report=report, manifest=manifest).results

    assert by_name(result.checks)["crashes"].verdict is Verdict.UNVERIFIABLE
    assert result.fault is None


def test_a_crash_in_the_headline_check_is_a_grader_fault():
    report = evaluate([Case(name="a", inputs={"x": 1})], crashes, is_even)
    manifest = MANIFEST.model_copy(update={"headline_check": "crashes"})

    (result,) = run_from_report(report=report, manifest=manifest).results

    assert result.fault is FaultClass.GRADER


def test_a_case_without_the_headline_check_raises():
    report = evaluate([Case(name="a", inputs={"x": 1})], is_even)
    manifest = MANIFEST.model_copy(update={"headline_check": "answer"})

    with pytest.raises(ValueError, match=r"case 'a' trial 1 has no headline check 'answer'"):
        run_from_report(report=report, manifest=manifest)


def test_a_task_that_raises_becomes_a_failing_task_check():
    report = evaluate([Case(name="a", inputs={"x": -1})], is_even)

    (result,) = run_from_report(report=report, manifest=MANIFEST).results

    assert len(result.checks) == 1
    (check,) = result.checks
    assert check.name == TASK_CHECK_NAME
    assert check.verdict is Verdict.FAIL
    assert check.detail == "RuntimeError: negative input"
    assert result.duration_s is None
    assert result.fault is None
    assert result.input_digest == input_digest({"x": -1})


def test_a_page_with_a_headline_check_builds_and_scores_a_crashed_task_as_a_fail(
    tmp_path: Path,
):
    report = evaluate(
        [Case(name="ok", inputs={"x": 1}), Case(name="bad", inputs={"x": -1})], answer
    )
    manifest = MANIFEST.model_copy(update={"headline_check": "answer"})
    write_run(root=tmp_path, run=run_from_report(report=report, manifest=manifest))

    page = build_page(root=tmp_path)

    row = re.search(r'<tr><th scope="row">All cases</th>(.*?)</tr>', page)
    assert row
    assert re.findall(r"<td[^>]*>(.*?)</td>", row.group(1))[:2] == ["2", "50.0%"]


def test_fault_of_classifies_a_task_that_raised():
    report = evaluate([Case(name="a", inputs={"x": -1})], is_even)

    (result,) = run_from_report(
        report=report, manifest=MANIFEST, fault_of=lambda _: FaultClass.INFRA
    ).results

    assert result.fault is FaultClass.INFRA


def test_fault_of_overrides_the_grader_default_only_when_it_returns_a_fault():
    report = evaluate([Case(name="a", inputs={"x": 1})], crashes, is_even)

    (overridden,) = run_from_report(
        report=report, manifest=MANIFEST, fault_of=lambda _: FaultClass.REFERENCE
    ).results
    (defaulted,) = run_from_report(
        report=report, manifest=MANIFEST, fault_of=lambda _: None
    ).results

    assert overridden.fault is FaultClass.REFERENCE
    assert defaulted.fault is FaultClass.GRADER


def test_successful_case_records_inputs_digest_duration_and_no_cost():
    report = evaluate([Case(name="a", inputs={"x": 1})], is_even)

    (result,) = run_from_report(report=report, manifest=MANIFEST).results

    assert result.trial == 1
    assert result.input_digest == input_digest({"x": 1})
    assert result.duration_s == report.cases[0].task_duration
    assert result.cost_usd is None


def test_input_digest_is_stable_across_dict_key_order():
    assert input_digest({"a": 1, "b": [1, {"c": 2, "d": 3}]}) == input_digest(
        {"b": [1, {"d": 3, "c": 2}], "a": 1}
    )


def test_input_digest_differs_for_different_inputs():
    digest = input_digest({"x": 1})

    assert digest.startswith("sha256:")
    assert digest != input_digest({"x": 2})


def test_input_digest_reads_whole_floats_as_ints_and_keeps_other_floats():
    assert input_digest({"x": 1.0}) == input_digest({"x": 1})
    assert input_digest({"x": 1.5}) != input_digest({"x": 1})
    assert input_digest({"x": True}) != input_digest({"x": 1})


def test_input_digest_refuses_a_non_finite_float():
    with pytest.raises(ValueError, match="JSON compliant"):
        input_digest({"x": float("inf")})


def test_tags_of_is_applied_to_cases_and_task_failures():
    report = evaluate(
        [
            Case(name="ok", inputs={"x": 1}, metadata={"family": "share"}),
            Case(name="broken", inputs={"x": -1}, metadata={"family": "total"}),
        ],
        is_even,
    )

    run = run_from_report(
        report=report,
        manifest=MANIFEST,
        tags_of=lambda case: (f"family:{case.metadata['family']}",),
    )

    assert {result.case_id: result.tags for result in run.results} == {
        "ok": ("family:share",),
        "broken": ("family:total",),
    }


def test_tags_that_differ_between_repeats_raise_before_the_run_is_saved():
    report = evaluate([Case(name="a", inputs={"x": 1})], is_even, repeat=2)
    trials = iter((("dac",), ()))

    with pytest.raises(ValueError, match="tags must be the same on every trial"):
        run_from_report(report=report, manifest=MANIFEST, tags_of=lambda _: next(trials))


def test_trace_of_records_the_system_trace_for_cases_and_task_failures():
    report = evaluate([Case(name="ok", inputs={"x": 1}), Case(name="broken", inputs={"x": -1})])

    run = run_from_report(
        report=report, manifest=MANIFEST, trace_of=lambda case: f"system-{case.name}"
    )

    assert {result.case_id: result.trace_id for result in run.results} == {
        "ok": "system-ok",
        "broken": "system-broken",
    }


def test_without_trace_of_no_trace_id_is_recorded():
    report = evaluate([Case(name="ok", inputs={"x": 1}), Case(name="broken", inputs={"x": -1})])

    run = run_from_report(report=report, manifest=MANIFEST)

    assert [result.trace_id for result in run.results] == [None, None]


def test_results_are_ordered_by_case_id_then_trial():
    report = evaluate(
        [
            Case(name="c", inputs={"x": 1}),
            Case(name="a", inputs={"x": -1}),
            Case(name="b", inputs={"x": 2}),
        ],
        is_even,
        repeat=2,
    )

    run = run_from_report(report=report, manifest=MANIFEST)

    assert [(result.case_id, result.trial) for result in run.results] == [
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
        ("c", 1),
        ("c", 2),
    ]


def test_a_report_level_evaluator_failure_raises_naming_the_evaluator():
    report = evaluate([Case(name="a", inputs={"x": 1})], crashes, is_even)
    report.report_evaluator_failures.append(report.cases[0].evaluator_failures.pop())

    with pytest.raises(ValueError, match=r"report-level evaluator output \['crashes'\]"):
        run_from_report(report=report, manifest=MANIFEST)


def test_a_duplicated_case_and_trial_raises():
    report = evaluate([Case(name="a", inputs={"x": 1})], is_even)
    report.cases.append(report.cases[0])

    with pytest.raises(ValueError, match="case 'a' trial 1 appears twice"):
        run_from_report(report=report, manifest=MANIFEST)


def test_a_case_without_a_name_raises():
    report = evaluate([Case(name="a", inputs={"x": 1})], is_even)
    report.cases[0].name = ""

    with pytest.raises(ValueError, match="has no name"):
        run_from_report(report=report, manifest=MANIFEST)


def test_the_run_round_trips_through_write_run_and_read_run(tmp_path):
    report = evaluate(
        [Case(name="a", inputs={"x": 1}), Case(name="b", inputs={"x": -1})],
        country_match,
        is_even,
        crashes,
    )
    run = run_from_report(report=report, manifest=MANIFEST, tags_of=lambda case: ("toy",))

    assert read_run(write_run(root=tmp_path, run=run)) == run
