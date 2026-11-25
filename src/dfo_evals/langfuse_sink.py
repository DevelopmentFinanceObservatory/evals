"""Push a run record to Langfuse as an experiment, a view of the records on disk.

A suite keeps its cases as Langfuse dataset items (``upsert_items``), then pushes each
run as one experiment over them (``push_run``) and reads it back (``verify_pushed``).
Langfuse runs nothing: the experiment's task returns the output the run already holds
and its evaluator returns the scores the run's verdicts already give.

Langfuse shows one item run per dataset item in an experiment, so a case has one item
run however many trials it ran. Its output holds every trial's result (and the caller's
output for that trial, if given) in trial order. Each check is scored once per case, as
the share of the case's trials whose verdict on it is a pass, over the trials that passed
or failed it: 1.0 or 0.0 for a case run once. Unverifiable, n/a and unreached verdicts
are left out, and a check no trial passed or failed gets no score. Each trial's number, fault, duration, cost, tags, input digest and
trace id are in the item run's metadata under ``trials``.

The experiment is named ``<suite>/<run_id> <revision> <digest>``, where ``<digest>``
begins the digest of the run's records and outputs. Pushing the same run again sends
only the cases and scores Langfuse lacks; a different run never shares a name.

Every digest here is taken after whole-number floats become integers, so a store that
returns ``1.0`` as ``1`` still matches.

langfuse 5 derives an experiment's id from the project, dataset and run name, so a
re-push lands in the same experiment. langfuse 4 takes it from the
``dataset_run_items.create`` endpoint and, when that call fails, falls back to a random
id per ``run_experiment`` call, so a re-push that sends missing cases can land under a
second experiment id with the same run name. Reading back by experiment name sees both.
"""

import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from langfuse import Evaluation, Langfuse
from langfuse.api import DatasetItem, DatasetStatus, NotFoundError
from langfuse.experiment import ExperimentItem
from opentelemetry.sdk.trace import TracerProvider
from pydantic_core import to_jsonable_python

from dfo_evals.records import CaseResult, Run, Verdict
from dfo_evals.report import DIGEST_PREFIX, input_digest

logger = logging.getLogger(__name__)

VERIFY_TIMEOUT_S: Final = 180.0
# How often verify_pushed reads Langfuse back while it waits.
_POLL_INTERVAL_S: Final = 5.0
# Item runs run_experiment sends at once.
_PUSH_CONCURRENCY: Final = 8
# Experiment items per page when reading back; the API caps a page at 100.
_PAGE_SIZE: Final = 100
# Hex characters of the records digest kept in the experiment run name.
_RUN_NAME_DIGEST_LENGTH: Final = 12
_SCORE_DATA_TYPE: Final = "NUMERIC"
_PASS_SCORE: Final = 1.0
_FAIL_SCORE: Final = 0.0
_SCORED_VERDICTS: Final = frozenset({Verdict.PASS, Verdict.FAIL})
# Langfuse reads a comma in an experiment name filter as a list of names.
_NAME_SEPARATOR: Final = ","
_READ_FIELDS: Final = "core,io,metadata,scores"
# Scores list_items returns per item run at most; the API caps score_limit at 50.
_MAX_SCORES: Final = 50
# Langfuse disables every client, whatever its settings, when this variable is "false".
_TRACING_ENABLED_ENV: Final = "LANGFUSE_TRACING_ENABLED"
_TRACING_DISABLED: Final = "false"

# Item-run metadata keys. Every value is a string, as Langfuse stores span attributes.
_CASE_ID_KEY: Final = "case_id"
_CASE_DIGEST_KEY: Final = "case_digest"
_OUTPUT_DIGEST_KEY: Final = "output_digest"
_TRIALS_KEY: Final = "trials"


class LangfuseSinkError(RuntimeError):
    """Base class for errors pushing a run to Langfuse."""


class WrongProjectError(LangfuseSinkError):
    """The Langfuse keys name a project other than the expected one."""


class UnknownCaseError(LangfuseSinkError):
    """The run holds a case with no item in the dataset it is pushed to."""


class PushMismatchError(LangfuseSinkError):
    """Langfuse holds something under the experiment that the run does not."""


class PushIncompleteError(LangfuseSinkError):
    """Cases sent to Langfuse never appeared there within the timeout."""


class PushTimeoutError(LangfuseSinkError):
    """Langfuse did not show the whole push before the timeout."""


@dataclass(frozen=True)
class DatasetItemSpec:
    """A dataset item as a suite defines it.

    Attributes:
        item_id: Stable item id, unique across every dataset in the project.
        input: The case's input, any JSON-serialisable value.
        expected_output: What the case expects, any JSON-serialisable value.
        metadata: Extra fields shown with the item, any JSON-serialisable value.
    """

    item_id: str
    input: object
    expected_output: object = None
    metadata: object = None


@dataclass(frozen=True)
class _ItemRun:
    """One case's item run as Langfuse shows it."""

    trace_id: str
    case_digest: str
    output: object
    output_digest: str
    scores: Mapping[str, float | None]


def connect(*, public_key: str, secret_key: str, base_url: str, project_id: str) -> Langfuse:
    """Create a Langfuse client for one project that exports only its own spans.

    The client sends on a tracer provider of its own, so spans from the system under
    test in the same process are not exported to Langfuse.

    Args:
        public_key: Langfuse public key.
        secret_key: Langfuse secret key.
        base_url: Langfuse API base URL, such as ``https://cloud.langfuse.com``.
        project_id: The project the keys must name (``check_project``).

    Returns:
        The client.

    Raises:
        ValueError: A credential or the project id is empty, or
            ``LANGFUSE_TRACING_ENABLED`` is ``false``, which stops the client sending.
        WrongProjectError: The keys name another project.
    """
    settings = {
        "public_key": public_key,
        "secret_key": secret_key,
        "base_url": base_url,
        "project_id": project_id,
    }
    if missing := [name for name, value in settings.items() if not value]:
        raise ValueError(f"Langfuse {', '.join(missing)} must not be empty")
    if os.environ.get(_TRACING_ENABLED_ENV, "").lower() == _TRACING_DISABLED:
        raise ValueError(f"{_TRACING_ENABLED_ENV} is false, so Langfuse would receive nothing")
    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        tracer_provider=TracerProvider(),
    )
    try:
        check_project(client=client, project_id=project_id)
    except WrongProjectError:
        client.shutdown()
        raise
    return client


def check_project(*, client: Langfuse, project_id: str) -> None:
    """Refuse a client whose keys name any project but ``project_id``.

    Args:
        client: The Langfuse client.
        project_id: The project the client must write to.

    Raises:
        WrongProjectError: The keys name another project, or more than one.
    """
    found = [project.id for project in client.api.projects.get().data]
    if found != [project_id]:
        # The other projects' ids stay out of the message, which public CI logs show.
        raise WrongProjectError(
            f"the Langfuse keys name {len(found)} project(s), not {project_id} alone"
        )


def upsert_items(*, client: Langfuse, dataset: str, items: Sequence[DatasetItemSpec]) -> list[str]:
    """Write each item to a dataset where Langfuse lacks it or holds it differently.

    Creates the dataset if it is missing. An archived item is written again as active.

    Args:
        client: The Langfuse client.
        dataset: Dataset name.
        items: The items the dataset must hold.

    Returns:
        The ids of the items written, in the order given.

    Raises:
        ValueError: Two items share an id, or an item has no input, which
            ``run_experiment`` would skip.
    """
    ids = [item.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"dataset {dataset!r}: item ids repeat: {ids}")
    if no_input := [item.item_id for item in items if item.input is None]:
        raise ValueError(f"dataset {dataset!r}: items have no input: {no_input}")

    held = _held_items(client=client, dataset=dataset, create=True)
    written = []
    for spec in items:
        input_, expected_output, metadata = (
            to_jsonable_python(spec.input),
            to_jsonable_python(spec.expected_output),
            to_jsonable_python(spec.metadata),
        )
        item = held.get(spec.item_id)
        if (
            item is not None
            and item.status == DatasetStatus.ACTIVE
            and (item.input, item.expected_output, item.metadata)
            == (input_, expected_output, metadata)
        ):
            continue
        client.create_dataset_item(
            dataset_name=dataset,
            id=spec.item_id,
            input=input_,
            expected_output=expected_output,
            metadata=metadata,
            status=DatasetStatus.ACTIVE,
        )
        written.append(spec.item_id)
    logger.info("dataset %s: wrote %d of %d items", dataset, len(written), len(items))
    return written


def push_run(
    *,
    client: Langfuse,
    run: Run,
    dataset: str,
    item_id_of: Callable[[str], str],
    outputs: Mapping[tuple[str, int], object] | None = None,
    timeout_s: float = VERIFY_TIMEOUT_S,
) -> str:
    """Push a run as one experiment over a dataset, sending only what Langfuse lacks.

    Every case of the run must already be an item of the dataset (``upsert_items``).
    After sending, Langfuse is read back until it shows every case and score, so a
    push that returns has been checked as ``verify_pushed`` checks it.

    Args:
        client: The Langfuse client.
        run: The run to push.
        dataset: Name of the dataset holding the run's cases.
        item_id_of: Maps a case id to its dataset item id.
        outputs: What the system under test returned, by case id and trial. A trial
            missing from it has no output.
        timeout_s: How long to wait for Langfuse to show the push.

    Returns:
        The experiment run name.

    Raises:
        UnknownCaseError: A case has no item in the dataset. Nothing is written.
        ValueError: ``outputs`` names a case and trial the run lacks, the revision holds
            a comma, a case scores more checks than Langfuse reads back, or a dataset
            item has no input. Nothing is written.
        PushMismatchError: Langfuse holds a case under this experiment that differs from
            the run's.
        PushIncompleteError: Cases sent never appeared within ``timeout_s``. Pushing
            the run again sends them.
        PushTimeoutError: Every case appeared but some scores did not within
            ``timeout_s``. Pushing the run again writes the missing scores.
    """
    outputs = outputs or {}
    if extra := sorted(set(outputs) - {(r.case_id, r.trial) for r in run.results}):
        raise ValueError(f"outputs name case trials the run lacks: {extra}")

    by_case = _results_by_case(run)
    held_items = _held_items(client=client, dataset=dataset, create=False)
    item_ids = {case_id: item_id_of(case_id) for case_id in by_case}
    if unknown := [case_id for case_id, item_id in item_ids.items() if item_id not in held_items]:
        raise UnknownCaseError(f"dataset {dataset!r} holds no item for cases {unknown}")
    if no_input := [i for i in item_ids.values() if held_items[i].input is None]:
        raise ValueError(f"dataset {dataset!r}: items have no input: {no_input}")
    expected_scores = {case_id: case_scores(results) for case_id, results in by_case.items()}
    if over := [c for c, scores in expected_scores.items() if len(scores) > _MAX_SCORES]:
        raise ValueError(
            f"cases {over} score more than {_MAX_SCORES} checks, more than Langfuse reads back"
        )

    run_name = experiment_run_name(run=run, outputs=outputs)
    case_outputs = {
        case_id: _case_output(case_id, results, outputs) for case_id, results in by_case.items()
    }

    held = _read_pushed(client=client, run=run, run_name=run_name)
    for case_id, item_run in held.items():
        if case_id not in by_case or item_run.case_digest != _case_digest(by_case[case_id]):
            raise PushMismatchError(
                f"Langfuse holds case {case_id!r} under {run_name} with results the run lacks"
            )
        if _digest(item_run.output) != _digest(case_outputs[case_id]):
            raise PushMismatchError(
                f"Langfuse holds case {case_id!r} under {run_name} with another output"
            )

    pending = [case_id for case_id in by_case if case_id not in held]
    if pending:
        _send(
            client=client,
            run=run,
            run_name=run_name,
            data=[
                held_items[item_ids[case_id]].model_copy(
                    update={"metadata": _item_run_metadata(by_case[case_id], case_outputs[case_id])}
                )
                for case_id in pending
            ],
            outputs={case_id: case_outputs[case_id] for case_id in pending},
            scores=expected_scores,
        )

    filled = 0
    for case_id, item_run in held.items():
        for evaluation in expected_scores[case_id]:
            if evaluation.name not in item_run.scores:
                client.create_score(
                    trace_id=item_run.trace_id,
                    name=evaluation.name,
                    value=float(evaluation.value),
                    data_type=_SCORE_DATA_TYPE,
                )
                filled += 1
    client.flush()
    logger.info(
        "%s: sent %d cases, %d already in Langfuse, filled %d missing scores",
        run_name,
        len(pending),
        len(held),
        filled,
    )

    shown, missing = _await_visible(client=client, run=run, run_name=run_name, timeout_s=timeout_s)
    if never := [case_id for case_id in pending if case_id not in shown]:
        raise PushIncompleteError(
            f"after {timeout_s:.0f} s Langfuse shows {len(pending) - len(never)} of the "
            f"{len(pending)} cases sent under {run_name}; push the run again to send "
            f"{never}"
        )
    if missing:
        raise PushTimeoutError(
            f"after {timeout_s:.0f} s Langfuse lacks, under {run_name}: {'; '.join(missing)}; "
            "push the run again to write the missing scores"
        )
    return run_name


def verify_pushed(
    *, client: Langfuse, run: Run, run_name: str, timeout_s: float = VERIFY_TIMEOUT_S
) -> None:
    """Wait until Langfuse shows every case and score of a pushed run, and check them.

    Args:
        client: The Langfuse client.
        run: The run that was pushed.
        run_name: The experiment run name ``push_run`` returned.
        timeout_s: How long to wait for Langfuse to show the whole push.

    Raises:
        PushMismatchError: Langfuse shows a case or score that differs from the run.
        PushTimeoutError: Langfuse still lacks a case or score after ``timeout_s``.
    """
    _, missing = _await_visible(client=client, run=run, run_name=run_name, timeout_s=timeout_s)
    if missing:
        # No re-push hint: a slow Langfuse would get the same cases and scores twice.
        raise PushTimeoutError(
            f"after {timeout_s:.0f} s Langfuse lacks, under {run_name}: {'; '.join(missing)}"
        )


def _await_visible(
    *, client: Langfuse, run: Run, run_name: str, timeout_s: float
) -> tuple[dict[str, _ItemRun], list[str]]:
    """Read Langfuse back until it shows every case and score of a run, or time runs out.

    Returns:
        The item runs Langfuse shows by case id, and a description of each case and
        score it still lacks, empty when it shows everything.

    Raises:
        PushMismatchError: Langfuse shows a case or score that differs from the run.
    """
    by_case = _results_by_case(run)
    expected_digests = {case_id: _case_digest(results) for case_id, results in by_case.items()}
    expected_scores = {
        case_id: {e.name: float(e.value) for e in case_scores(results)}
        for case_id, results in by_case.items()
    }

    deadline = time.monotonic() + timeout_s
    while True:
        held = _read_pushed(client=client, run=run, run_name=run_name)
        _check_matches(
            held=held, digests=expected_digests, scores=expected_scores, run_name=run_name
        )
        missing = _missing(held=held, scores=expected_scores)
        if not missing:
            logger.info("%s: Langfuse shows all %d cases", run_name, len(by_case))
            return held, []
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return held, missing
        time.sleep(min(_POLL_INTERVAL_S, remaining))


def experiment_run_name(*, run: Run, outputs: Mapping[tuple[str, int], object]) -> str:
    """Name a run's experiment ``<suite>/<run_id> <revision> <digest>``.

    Args:
        run: The run.
        outputs: The outputs pushed with it, by case id and trial.

    Returns:
        The name, whose digest begins the digest of the run's records and outputs.

    Raises:
        ValueError: The revision holds a comma, which Langfuse reads as a list of names.
    """
    manifest = run.manifest
    if _NAME_SEPARATOR in manifest.revision:
        raise ValueError(f"revision {manifest.revision!r} holds a comma")
    digest = _digest(
        {
            "run": run.model_dump(mode="json"),
            "outputs": [
                [case_id, trial, to_jsonable_python(output)]
                for (case_id, trial), output in sorted(outputs.items())
            ],
        }
    )
    short = digest.removeprefix(DIGEST_PREFIX)[:_RUN_NAME_DIGEST_LENGTH]
    return f"{manifest.suite}/{manifest.run_id} {manifest.revision} {short}"


def case_scores(results: Sequence[CaseResult]) -> list[Evaluation]:
    """Score each check of one case over its trials.

    Args:
        results: Every trial of one case.

    Returns:
        One score per check that some trial passed or failed: the share of those
        trials that passed. A check no trial passed or failed has no score.
    """
    values: defaultdict[str, list[float]] = defaultdict(list)
    for result in results:
        for check in result.checks:
            if check.verdict in _SCORED_VERDICTS:
                score = _PASS_SCORE if check.verdict is Verdict.PASS else _FAIL_SCORE
                values[check.name].append(score)
    return [
        Evaluation(name=name, value=sum(scores) / len(scores), data_type=_SCORE_DATA_TYPE)
        for name, scores in values.items()
    ]


def _send(
    *,
    client: Langfuse,
    run: Run,
    run_name: str,
    data: list[DatasetItem],
    outputs: Mapping[str, object],
    scores: Mapping[str, list[Evaluation]],
) -> None:
    """Send one item run per dataset item through ``run_experiment``."""

    def task(*, item: ExperimentItem, **_: object) -> object:
        if not isinstance(item, DatasetItem):
            raise TypeError(f"expected a DatasetItem, got {type(item).__name__}")
        return outputs[_case_id_of(item.metadata)]

    def evaluate(*, metadata: object = None, **_: object) -> list[Evaluation]:
        return scores[_case_id_of(metadata)]

    manifest = run.manifest
    result = client.run_experiment(
        name=manifest.suite,
        run_name=run_name,
        description=f"{manifest.suite} run {manifest.run_id} at {manifest.revision}",
        data=data,
        task=task,
        evaluators=[evaluate],
        max_concurrency=_PUSH_CONCURRENCY,
        metadata={
            "suite": manifest.suite,
            "run_id": manifest.run_id,
            "revision": manifest.revision,
            "dataset_digest": manifest.dataset_digest,
            "started_at": manifest.started_at.isoformat(),
        },
    )
    if len(result.item_results) != len(data):
        # The read-back after sending decides which cases are truly missing.
        logger.warning(
            "%s: run_experiment returned %d of %d item runs",
            run_name,
            len(result.item_results),
            len(data),
        )


def _read_pushed(*, client: Langfuse, run: Run, run_name: str) -> dict[str, _ItemRun]:
    """Read the item runs Langfuse shows under an experiment, by case id, newest first.

    Raises:
        PushMismatchError: An item run lacks the metadata a push writes.
    """
    found: dict[str, _ItemRun] = {}
    cursor: str | None = None
    while True:
        page = client.api.experiments.list_items(
            # No item run of this run's push can start before the run did.
            from_start_time=run.manifest.started_at,
            experiment_name=run_name,
            fields=_READ_FIELDS,
            limit=_PAGE_SIZE,
            score_limit=_MAX_SCORES,
            cursor=cursor,
        )
        for item in page.data:
            metadata = item.metadata or {}
            keys = (_CASE_ID_KEY, _CASE_DIGEST_KEY, _OUTPUT_DIGEST_KEY)
            if any(key not in metadata for key in keys):
                raise PushMismatchError(
                    f"item run {item.id} under {run_name} lacks the metadata a push writes"
                )
            # Newest first, so a case sent twice reads as its latest item run.
            found.setdefault(
                str(metadata[_CASE_ID_KEY]),
                _ItemRun(
                    trace_id=item.trace_id,
                    case_digest=str(metadata[_CASE_DIGEST_KEY]),
                    output=_parsed(item.output),
                    output_digest=str(metadata[_OUTPUT_DIGEST_KEY]),
                    scores={
                        score.name: _numeric(getattr(score, "value", None))
                        for score in item.scores or ()
                    },
                ),
            )
        cursor = page.meta.cursor
        if not page.data or not cursor:
            return found


def _check_matches(
    *,
    held: Mapping[str, _ItemRun],
    digests: Mapping[str, str],
    scores: Mapping[str, Mapping[str, float]],
    run_name: str,
) -> None:
    """Raise if anything Langfuse shows differs from the run.

    Raises:
        PushMismatchError: A case is not in the run, its records or output differ, or a
            score has another value.
    """
    for case_id, item_run in held.items():
        if case_id not in digests:
            raise PushMismatchError(f"Langfuse shows case {case_id!r} under {run_name}, not in run")
        if item_run.case_digest != digests[case_id]:
            raise PushMismatchError(f"case {case_id!r} under {run_name}: records differ")
        output = item_run.output
        if _digest(output) != item_run.output_digest or not isinstance(output, dict):
            raise PushMismatchError(f"case {case_id!r} under {run_name}: output differs")
        if _digest(output.get("results")) != digests[case_id]:
            raise PushMismatchError(f"case {case_id!r} under {run_name}: output results differ")
        for name, value in item_run.scores.items():
            if name in scores[case_id] and value != scores[case_id][name]:
                raise PushMismatchError(
                    f"case {case_id!r} under {run_name}: score {name!r} is {value}, "
                    f"not {scores[case_id][name]}"
                )


def _missing(
    *, held: Mapping[str, _ItemRun], scores: Mapping[str, Mapping[str, float]]
) -> list[str]:
    """Describe each case and score Langfuse does not show yet."""
    missing = []
    for case_id, wanted in scores.items():
        if case_id not in held:
            missing.append(f"case {case_id!r}")
        elif lacking := sorted(set(wanted) - set(held[case_id].scores)):
            missing.append(f"case {case_id!r} scores {lacking}")
    return missing


def _held_items(*, client: Langfuse, dataset: str, create: bool) -> dict[str, DatasetItem]:
    """Return a dataset's items by id; a missing dataset is created or read as empty."""
    try:
        return {item.id: item for item in client.get_dataset(dataset).items}
    except NotFoundError:
        if create:
            client.create_dataset(name=dataset)
            logger.info("created dataset %s", dataset)
        return {}


def _results_by_case(run: Run) -> dict[str, list[CaseResult]]:
    """Group a run's results by case id, each case's trials in order."""
    by_case: defaultdict[str, list[CaseResult]] = defaultdict(list)
    for result in sorted(run.results, key=lambda result: (result.case_id, result.trial)):
        by_case[result.case_id].append(result)
    return dict(by_case)


def _case_output(
    case_id: str, results: Sequence[CaseResult], outputs: Mapping[tuple[str, int], object]
) -> dict[str, object]:
    """The output of a case's item run: every trial's result and output."""
    return {
        "case_id": case_id,
        "results": _case_records(results),
        "outputs": [to_jsonable_python(outputs.get((case_id, result.trial))) for result in results],
    }


def _item_run_metadata(results: Sequence[CaseResult], output: object) -> dict[str, str]:
    """The metadata of a case's item run, every value a string."""
    trials = [
        {
            "trial": result.trial,
            "fault": result.fault,
            "duration_s": result.duration_s,
            "cost_usd": result.cost_usd,
            "tags": list(result.tags),
            "input_digest": result.input_digest,
            "trace_id": result.trace_id,
        }
        for result in results
    ]
    return {
        _CASE_ID_KEY: results[0].case_id,
        _CASE_DIGEST_KEY: _case_digest(results),
        _OUTPUT_DIGEST_KEY: _digest(output),
        _TRIALS_KEY: json.dumps(trials, ensure_ascii=False),
    }


def _case_records(results: Sequence[CaseResult]) -> list[object]:
    """A case's results as JSON-compatible Python."""
    return [result.model_dump(mode="json") for result in results]


def _case_digest(results: Sequence[CaseResult]) -> str:
    """Digest a case's results, as its item run's output holds them."""
    return _digest(_case_records(results))


def _digest(value: object) -> str:
    """Digest any JSON-serialisable value by its canonical JSON, whole floats as ints."""
    # input_digest is a canonical JSON digest of any value, not only inputs.
    return input_digest(_whole_floats_as_ints(to_jsonable_python(value)))


def _whole_floats_as_ints(value: object) -> object:
    """Turn every whole-number float into an int, as JSON read back from Langfuse has it."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _whole_floats_as_ints(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_whole_floats_as_ints(item) for item in value]
    return value


def _parsed(output: object) -> object:
    """Read an item run's output, which Langfuse may return as JSON text."""
    if not isinstance(output, str):
        return output
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return output


def _numeric(value: object) -> float | None:
    """A score's value as a float, or None when it is not numeric."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _case_id_of(metadata: object) -> str:
    """The case id a push wrote into an item's metadata."""
    case_id = metadata.get(_CASE_ID_KEY) if isinstance(metadata, dict) else None
    if case_id is None:
        raise PushMismatchError(f"item metadata lacks {_CASE_ID_KEY!r}: {metadata!r}")
    return str(case_id)
