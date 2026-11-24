"""Convert a Pydantic Evals report into a run record.

Trials come from the report itself: ``Dataset.evaluate(repeat=n)`` names each run
``"<case> [<i>/<n>]"`` and sets ``source_case_name``, so the case id is the source case
name and the trial is ``i``. A report evaluated without ``repeat`` holds trial 1 only.
"""

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any, Final

from pydantic_core import to_jsonable_python
from pydantic_evals.reporting import EvaluationReport, ReportCase, ReportCaseFailure

from dfo_evals.records import (
    TASK_CHECK_NAME,
    CaseResult,
    CheckResult,
    FaultClass,
    Run,
    RunManifest,
    Verdict,
)

DIGEST_PREFIX: Final = "sha256:"
_SINGLE_TRIAL: Final = 1
_VERDICT_VALUES: Final = frozenset(item.value for item in Verdict)

type AnyReportCase = ReportCase[Any, Any, Any]
type AnyReportCaseFailure = ReportCaseFailure[Any, Any, Any]
# Failures carry the same inputs and metadata, so tags can be derived for both.
type TagsOf = Callable[[AnyReportCase | AnyReportCaseFailure], tuple[str, ...]]
type FaultOf = Callable[[AnyReportCase | AnyReportCaseFailure], FaultClass | None]


def input_digest(inputs: object) -> str:
    """Digest case inputs so a result can be matched to the exact inputs it ran on.

    The inputs are converted to JSON-compatible Python, then serialised with sorted keys
    and no whitespace, so dict key order does not change the digest. Sets serialise in
    iteration order, which can differ between processes; pass sorted sequences instead.

    Args:
        inputs: The case inputs, any value Pydantic can serialise.

    Returns:
        ``"sha256:"`` followed by the hex digest of the canonical JSON.

    Raises:
        ValueError: The inputs hold a non-finite float.
    """
    canonical = json.dumps(
        to_jsonable_python(inputs),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return DIGEST_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_from_report(
    *,
    report: EvaluationReport[Any, Any, Any],
    manifest: RunManifest,
    tags_of: TagsOf | None = None,
    fault_of: FaultOf | None = None,
) -> Run:
    """Convert an evaluation report into a run with one result per case and trial.

    Labels become their verdicts, assertions become pass or fail, and an evaluator that
    raised becomes ``unverifiable`` and a task that raised becomes one ``task`` check that
    fails, with no duration.

    A case's fault is what ``fault_of`` returns for it. When ``fault_of`` is None or
    returns None, a case with an evaluator that raised is a grader fault and any other
    case has no fault.

    Args:
        report: The report ``Dataset.evaluate`` returned.
        manifest: Metadata of the run that produced the report.
        tags_of: Derives a case's tags from its report case or task failure. None
            gives every case no tags.
        fault_of: Classifies who is at fault for a case or task failure, such as
            ``INFRA`` for a timeout.

    Returns:
        The run, with results ordered by case id then trial.

    Raises:
        ValueError: The report holds output from report-level evaluators, a case has no
            name, a case and trial appear twice, a label is not a verdict, or an
            evaluator returned a numeric score.
    """
    _reject_report_evaluator_output(report)
    results = [_result_from_case(case, tags_of=tags_of, fault_of=fault_of) for case in report.cases]
    results += [
        _result_from_failure(failure, tags_of=tags_of, fault_of=fault_of)
        for failure in report.failures
    ]

    seen: set[tuple[str, int]] = set()
    for result in results:
        key = (result.case_id, result.trial)
        if key in seen:
            raise ValueError(f"case {result.case_id!r} trial {result.trial} appears twice")
        seen.add(key)

    results.sort(key=lambda result: (result.case_id, result.trial))
    return Run(manifest=manifest, results=tuple(results))


def _reject_report_evaluator_output(report: EvaluationReport[Any, Any, Any]) -> None:
    """Raise if report-level evaluators produced output, which run records cannot hold.

    Raises:
        ValueError: The report holds analyses or report-level evaluator failures.
    """
    names = [analysis.title for analysis in report.analyses] + [
        failure.name for failure in report.report_evaluator_failures
    ]
    if names:
        raise ValueError(
            f"report-level evaluator output {names} has no place in a run record; "
            "write checks as case evaluators"
        )


def _result_from_case(
    case: AnyReportCase, *, tags_of: TagsOf | None, fault_of: FaultOf | None
) -> CaseResult:
    """Convert one successful report case into a case result."""
    case_id, trial = _case_id_and_trial(name=case.name, source_case_name=case.source_case_name)
    if case.scores:
        raise ValueError(
            f"case {case.name!r}: evaluators {sorted(case.scores)} returned numeric "
            "scores; checks must return a verdict label or a bool"
        )

    checks: list[CheckResult] = []
    for name, label in case.labels.items():
        if label.value not in _VERDICT_VALUES:
            raise ValueError(
                f"case {case.name!r}: evaluator {name!r} returned label {label.value!r}, "
                f"which is not a verdict ({sorted(_VERDICT_VALUES)})"
            )
        checks.append(CheckResult(name=name, verdict=Verdict(label.value), detail=label.reason))
    checks.extend(
        CheckResult(
            name=name,
            verdict=Verdict.PASS if assertion.value else Verdict.FAIL,
            detail=assertion.reason,
        )
        for name, assertion in case.assertions.items()
    )
    checks.extend(
        CheckResult(name=failure.name, verdict=Verdict.UNVERIFIABLE, detail=failure.error_message)
        for failure in case.evaluator_failures
    )

    return CaseResult(
        case_id=case_id,
        trial=trial,
        input_digest=input_digest(case.inputs),
        checks=tuple(checks),
        fault=_fault(
            case,
            fault_of=fault_of,
            default=FaultClass.GRADER if case.evaluator_failures else None,
        ),
        duration_s=case.task_duration,
        cost_usd=None,
        trace_id=case.trace_id,
        tags=tags_of(case) if tags_of is not None else (),
    )


def _result_from_failure(
    failure: AnyReportCaseFailure, *, tags_of: TagsOf | None, fault_of: FaultOf | None
) -> CaseResult:
    """Convert a task that raised into a case result with one failing ``task`` check."""
    case_id, trial = _case_id_and_trial(
        name=failure.name, source_case_name=failure.source_case_name
    )
    return CaseResult(
        case_id=case_id,
        trial=trial,
        input_digest=input_digest(failure.inputs),
        checks=(
            CheckResult(name=TASK_CHECK_NAME, verdict=Verdict.FAIL, detail=failure.error_message),
        ),
        fault=_fault(failure, fault_of=fault_of, default=None),
        duration_s=None,
        cost_usd=None,
        trace_id=failure.trace_id,
        tags=tags_of(failure) if tags_of is not None else (),
    )


def _fault(
    item: AnyReportCase | AnyReportCaseFailure,
    *,
    fault_of: FaultOf | None,
    default: FaultClass | None,
) -> FaultClass | None:
    """Return the caller's fault for a case, or the default when it gives none."""
    fault = fault_of(item) if fault_of is not None else None
    return fault if fault is not None else default


def _case_id_and_trial(*, name: str, source_case_name: str | None) -> tuple[str, int]:
    """Split a report case name into case id and trial.

    Raises:
        ValueError: The case has no name, or a repeated run's name lacks its run index.
    """
    if source_case_name is None:
        if not name:
            raise ValueError("report case has no name")
        return name, _SINGLE_TRIAL
    if not source_case_name:
        raise ValueError(f"report case {name!r} has an empty source case name")
    match = re.fullmatch(rf"{re.escape(source_case_name)} \[(\d+)/\d+\]", name)
    if match is None:
        raise ValueError(
            f"report case {name!r} does not carry a run index for source case {source_case_name!r}"
        )
    return source_case_name, int(match.group(1))
