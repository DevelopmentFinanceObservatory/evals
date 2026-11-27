"""Push a run record to Langfuse as an experiment, a view of the records on disk.

A suite keeps its cases as Langfuse dataset items (``upsert_items``), then pushes each
run as one experiment over them (``push_run``) and reads it back (``verify_pushed``).
The caller maps each case to its dataset and item (``DatasetItemRef``), so one run may
span several datasets. The run then lands as one experiment in each of those datasets,
all under the same run name, and every dataset keeps its own item ids and run history.
Langfuse runs nothing: the experiment's task returns the output the run already holds.

A dataset item's input must be the case's inputs exactly as the run evaluated them:
``input_digest`` of the item's input must equal every trial's stored ``input_digest``.
A push or verification over an item whose input differs refuses before writing, and
each item run read back must be of the item its case maps to, in that item's dataset,
and must hold that same input.

Langfuse shows one item run per dataset item in an experiment, so a case has one item
run however many trials it ran. Its output holds every trial's result (and the caller's
output for that trial, if given) in trial order. Each trial's number, fault, duration,
cost, tags, input digest and trace id are in the item run's metadata under ``trials``.

Each check is scored once per case, as the share of the case's trials whose verdict on
it is a pass, over the trials that passed or failed it: 1.0 or 0.0 for a case run once.
Unverifiable, n/a and unreached verdicts are left out, and a check no trial passed or
failed gets no score. A push sends the item runs with no evaluator, waits to see them,
then writes each score with ``create_score`` on the item run's trace and observation.
The score's id comes from the run name, dataset, item and check, and its timestamp is
the item run's start time, so a score written again overwrites the first.

The experiment is named ``<suite>/<run_id> <revision> <digest>``, where ``<digest>``
begins the digest of the run's records and outputs. The push passes the experiment id
Langfuse 5 derives from the project, dataset and run name, so pushing the same run again
lands in the same experiment and sends only the cases and scores Langfuse lacks. A different run never
shares a name. Every digest here is taken after whole-number floats become integers, so
a store that returns ``1.0`` as ``1`` still matches.

The run name's format, the item run metadata keys and the namespaces the experiment and
score ids derive from are fixed. Experiments already in Langfuse were written under
them, and a change would make a later push of the same run create a second experiment
and second scores.
"""

import hashlib
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

try:
    from langfuse import Evaluation, Langfuse
    from langfuse.api import DatasetItem, DatasetStatus, NotFoundError
    from langfuse.experiment import ExperimentItem
    from opentelemetry.sdk.trace import TracerProvider
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        f"dfo_evals.langfuse_sink needs the langfuse extra (dfo-evals[langfuse]): {error}"
    ) from error
from pydantic_core import to_jsonable_python

from dfo_evals.records import CaseResult, Run, Verdict
from dfo_evals.report import DIGEST_PREFIX, input_digest

logger = logging.getLogger(__name__)

VERIFY_TIMEOUT_S: Final = 180.0
# How often verify_pushed reads Langfuse back while it waits.
_POLL_INTERVAL_S: Final = 5.0
# Langfuse 5's experiment id: the first 16 hex characters of a SHA-256 over this
# namespace, the project, the dataset and the run name.
_EXPERIMENT_ID_NAMESPACE: Final = "langfuse-experiment-v1"
_EXPERIMENT_ID_LENGTH: Final = 16
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
# Namespace of the ids of the scores a push writes.
_SCORE_ID_NAMESPACE: Final = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://github.com/DevelopmentFinanceObservatory/evals/langfuse-score"
)


class LangfuseSinkError(RuntimeError):
    """Base class for errors pushing a run to Langfuse."""


class WrongProjectError(LangfuseSinkError):
    """The Langfuse keys name a project other than the expected one."""


class UnknownCaseError(LangfuseSinkError):
    """The run holds a case with no item in the dataset it maps to."""


class InputMismatchError(LangfuseSinkError):
    """A dataset item's input is not the input the run evaluated its case on."""


class PushMismatchError(LangfuseSinkError):
    """Langfuse holds something under the experiment that the run does not."""


class PushIncompleteError(LangfuseSinkError):
    """Cases sent to Langfuse never appeared there within the timeout."""


class PushTimeoutError(LangfuseSinkError):
    """Langfuse did not show the whole push before the timeout."""


@dataclass(frozen=True)
class DatasetItemSpec:
    """A dataset item as a suite defines it.

    ``None`` in ``expected_output`` or ``metadata`` keeps the value Langfuse holds: its
    public API reads a null field as absent and keeps the stored value, so no value
    clears one. To replace a stored value, give another, such as ``{}``.

    Attributes:
        item_id: Stable item id, unique across every dataset in the project.
        input: The case's input, any JSON-serialisable value but ``None``.
        expected_output: What the case expects, any JSON-serialisable value, or
            ``None`` to keep what Langfuse holds.
        metadata: Extra fields shown with the item, any JSON-serialisable value, or
            ``None`` to keep what Langfuse holds.
    """

    item_id: str
    input: object
    expected_output: object = None
    metadata: object = None


@dataclass(frozen=True)
class DatasetItemRef:
    """Where a case lives in Langfuse.

    Attributes:
        dataset: Name of the dataset holding the case.
        item_id: The case's dataset item id.
    """

    dataset: str
    item_id: str


@dataclass(frozen=True)
class _HeldDataset:
    """A dataset as Langfuse holds it."""

    dataset_id: str
    items: Mapping[str, DatasetItem]


@dataclass(frozen=True)
class _DatasetCases:
    """The run's cases that one dataset holds, each with its item."""

    dataset: str
    dataset_id: str
    items: Mapping[str, DatasetItem]


@dataclass(frozen=True)
class _ItemRun:
    """One case's item run as Langfuse shows it."""

    trace_id: str
    observation_id: str
    start_time: datetime
    item_id: str
    input: object
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


def _sole_project_id(client: Langfuse) -> str:
    """Return the one project the client's keys name.

    Raises:
        WrongProjectError: The keys name no project or more than one.
    """
    found = [project.id for project in client.api.projects.get().data]
    if len(found) != 1:
        raise WrongProjectError(f"the Langfuse keys name {len(found)} projects, not one")
    return found[0]


def _experiment_id(*, project_id: str, dataset_id: str, run_name: str) -> str:
    """Derive an experiment's id as Langfuse 5 does for a run on a dataset.

    Langfuse derives the same id when the caller gives none, but falls back to a random
    one if it cannot look up the project, which would split a re-push across two
    experiments. Passing it keeps every push of a run on a dataset in one experiment.
    """
    payload = json.dumps(
        [_EXPERIMENT_ID_NAMESPACE, project_id, dataset_id, run_name],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_EXPERIMENT_ID_LENGTH]


def upsert_items(*, client: Langfuse, dataset: str, items: Sequence[DatasetItemSpec]) -> list[str]:
    """Write each item to a dataset where Langfuse lacks it or holds it differently.

    Creates the dataset if it is missing. Langfuse's ``get_dataset`` returns active items
    only, so an archived item reads as missing and is written again as active.
    A ``None`` field keeps the stored value, so it never counts as a difference.
    Values compare by digest, whole-number floats as ints, as Langfuse may return them.

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

    held = _held_dataset(client=client, dataset=dataset, create=True)
    held_items = held.items if held else {}
    written = []
    for spec in items:
        wanted = {
            "input": to_jsonable_python(spec.input),
            "expected_output": to_jsonable_python(spec.expected_output),
            "metadata": to_jsonable_python(spec.metadata),
        }
        item = held_items.get(spec.item_id)
        if (
            item is not None
            and item.status == DatasetStatus.ACTIVE
            and not _changed_fields(held=item, wanted=wanted)
        ):
            continue
        client.create_dataset_item(
            dataset_name=dataset,
            id=spec.item_id,
            input=wanted["input"],
            expected_output=wanted["expected_output"],
            metadata=wanted["metadata"],
            status=DatasetStatus.ACTIVE,
        )
        written.append(spec.item_id)
    logger.info("dataset %s: wrote %d of %d items", dataset, len(written), len(items))
    return written


def push_run(
    *,
    client: Langfuse,
    run: Run,
    item_of: Callable[[str], DatasetItemRef],
    outputs: Mapping[tuple[str, int], object] | None = None,
    timeout_s: float = VERIFY_TIMEOUT_S,
) -> str:
    """Push a run as one experiment, sending only what Langfuse lacks.

    Every case of the run must already be an item of the dataset ``item_of`` maps it to
    (``upsert_items``), and that item's input must be the case's inputs exactly as the
    run evaluated them: its ``input_digest`` must equal the case's stored
    ``input_digest``. Item runs already in Langfuse count as pushed only when they are
    of the mapped item in its dataset and hold that input. The run lands as one
    experiment in each dataset its cases map to, all under one run name. After sending,
    Langfuse is read back until it shows every case, each case's scores are written
    under stable ids and its item run's start time, and Langfuse is read back again
    until it shows every score, so a push that returns has been checked as
    ``verify_pushed`` checks it. Writing a score again overwrites it, so pushing again
    never duplicates one.

    Args:
        client: The Langfuse client.
        run: The run to push.
        item_of: Maps a case id to its dataset and item.
        outputs: What the system under test returned, by case id and trial. A trial
            missing from it has no output.
        timeout_s: How long to wait for Langfuse to show the push, cases and scores
            together.

    Returns:
        The experiment run name, shared by every dataset's experiment.

    Raises:
        UnknownCaseError: A case has no item in the dataset it maps to. Nothing is
            written.
        InputMismatchError: A case's item input is not the input the run evaluated.
            Nothing is written.
        ValueError: ``outputs`` names a case and trial the run lacks, a case's trials
            hold different input digests, two cases map to one item, the revision holds
            a comma, a case scores more checks than Langfuse reads back, or a dataset
            item has no input. Nothing is written.
        PushMismatchError: Langfuse holds a case under this experiment that differs from
            the run's, in a dataset or item the case does not map to, or on another
            input.
        PushIncompleteError: Cases sent never appeared within ``timeout_s``. Pushing
            the run again sends them and writes their scores.
        PushTimeoutError: Every case appeared but some scores did not within
            ``timeout_s``. Pushing the run again writes those scores again under the
            same ids and timestamps, so a score that was only slow is overwritten, not
            duplicated.
    """
    outputs = outputs or {}
    if extra := sorted(set(outputs) - {(r.case_id, r.trial) for r in run.results}):
        raise ValueError(f"outputs name case trials the run lacks: {extra}")

    by_case = _results_by_case(run)
    input_digests = _case_input_digests(by_case)
    groups = _group_cases(client=client, case_ids=list(by_case), item_of=item_of)
    for group in groups:
        if no_input := [item.id for item in group.items.values() if item.input is None]:
            raise ValueError(f"dataset {group.dataset!r}: items have no input: {no_input}")
    _check_inputs(groups=groups, input_digests=input_digests)
    expected_scores = {case_id: case_scores(results) for case_id, results in by_case.items()}
    if over := [c for c, scores in expected_scores.items() if len(scores) > _MAX_SCORES]:
        raise ValueError(
            f"cases {over} score more than {_MAX_SCORES} checks, more than Langfuse reads back"
        )

    run_name = experiment_run_name(run=run, outputs=outputs)
    case_outputs = {
        case_id: _case_output(case_id, results, outputs) for case_id, results in by_case.items()
    }

    project_id = _sole_project_id(client)
    held = _read_pushed(
        client=client,
        run=run,
        run_name=run_name,
        groups=groups,
        project_id=project_id,
        input_digests=input_digests,
    )
    for case_id, item_run in held.items():
        if item_run.case_digest != _case_digest(by_case[case_id]):
            raise PushMismatchError(
                f"Langfuse holds case {case_id!r} under {run_name} with results the run lacks"
            )
        if _digest(item_run.output) != _digest(case_outputs[case_id]):
            raise PushMismatchError(
                f"Langfuse holds case {case_id!r} under {run_name} with another output"
            )

    pending = [case_id for case_id in by_case if case_id not in held]
    for group in groups:
        if not (sending := [case_id for case_id in group.items if case_id not in held]):
            continue
        # One run_experiment per dataset, since its experiment id names one dataset.
        _send(
            client=client,
            run=run,
            run_name=run_name,
            experiment_id=_experiment_id(
                project_id=project_id, dataset_id=group.dataset_id, run_name=run_name
            ),
            data=[
                group.items[case_id].model_copy(
                    update={"metadata": _item_run_metadata(by_case[case_id], case_outputs[case_id])}
                )
                for case_id in sending
            ],
            outputs={case_id: case_outputs[case_id] for case_id in sending},
        )

    dataset_of = {case_id: group.dataset for group in groups for case_id in group.items}
    written = _write_scores(
        client=client,
        run_name=run_name,
        item_runs=held,
        dataset_of=dataset_of,
        scores=expected_scores,
    )
    # Scores need the trace and start time of each item run sent, so wait to see them.
    deadline = time.monotonic() + timeout_s
    shown = held
    if pending:
        shown, _ = _await_visible(
            client=client,
            run=run,
            run_name=run_name,
            groups=groups,
            project_id=project_id,
            input_digests=input_digests,
            timeout_s=timeout_s,
            cases_only=True,
        )
        written += _write_scores(
            client=client,
            run_name=run_name,
            item_runs={case_id: shown[case_id] for case_id in pending if case_id in shown},
            dataset_of=dataset_of,
            scores=expected_scores,
        )
    client.flush()
    logger.info(
        "%s: sent %d cases, %d already in Langfuse, wrote %d scores, over %d datasets",
        run_name,
        len(pending),
        len(held),
        written,
        len(groups),
    )
    if never := [case_id for case_id in pending if case_id not in shown]:
        raise PushIncompleteError(
            f"after {timeout_s:.0f} s Langfuse shows {len(pending) - len(never)} of the "
            f"{len(pending)} cases sent under {run_name}; push the run again to send "
            f"{never}"
        )

    _, missing = _await_visible(
        client=client,
        run=run,
        run_name=run_name,
        groups=groups,
        project_id=project_id,
        input_digests=input_digests,
        timeout_s=max(0.0, deadline - time.monotonic()),
    )
    if missing:
        raise PushTimeoutError(
            f"after {timeout_s:.0f} s Langfuse lacks, under {run_name}: {'; '.join(missing)}; "
            "pushing the run again writes them again under the same ids, so a score that "
            "was only slow is overwritten, not duplicated"
        )
    return run_name


def verify_pushed(
    *,
    client: Langfuse,
    run: Run,
    run_name: str,
    item_of: Callable[[str], DatasetItemRef],
    timeout_s: float = VERIFY_TIMEOUT_S,
) -> None:
    """Wait until Langfuse shows every case and score of a pushed run, and check them.

    Each dataset the run's cases map to is read back under ``run_name``. As for
    ``push_run``, each case's dataset item must hold the input the run evaluated, and
    each item run must be of the mapped item and hold that input.

    Args:
        client: The Langfuse client.
        run: The run that was pushed.
        run_name: The experiment run name ``push_run`` returned.
        item_of: Maps a case id to its dataset and item, as given to ``push_run``.
        timeout_s: How long to wait for Langfuse to show the whole push.

    Raises:
        UnknownCaseError: A case has no item in the dataset it maps to.
        WrongProjectError: The client's keys name no project or more than one.
        InputMismatchError: A case's item input is not the input the run evaluated.
        ValueError: Two cases map to one item, or a case's trials hold different input
            digests.
        PushMismatchError: Langfuse shows a case or score that differs from the run, or
            shows a case in a dataset or item it does not map to, or on another input.
        PushTimeoutError: Langfuse still lacks a case or score after ``timeout_s``.
    """
    by_case = _results_by_case(run)
    input_digests = _case_input_digests(by_case)
    groups = _group_cases(client=client, case_ids=list(by_case), item_of=item_of)
    _check_inputs(groups=groups, input_digests=input_digests)
    project_id = _sole_project_id(client)
    _, missing = _await_visible(
        client=client,
        run=run,
        run_name=run_name,
        groups=groups,
        project_id=project_id,
        input_digests=input_digests,
        timeout_s=timeout_s,
    )
    if missing:
        # No re-push hint: a slow Langfuse would get the same cases and scores twice.
        raise PushTimeoutError(
            f"after {timeout_s:.0f} s Langfuse lacks, under {run_name}: {'; '.join(missing)}"
        )


def _await_visible(
    *,
    client: Langfuse,
    run: Run,
    run_name: str,
    groups: Sequence[_DatasetCases],
    project_id: str,
    input_digests: Mapping[str, str],
    timeout_s: float,
    cases_only: bool = False,
) -> tuple[dict[str, _ItemRun], list[str]]:
    """Read Langfuse back until it shows every case and score of a run, or time runs out.

    With ``cases_only``, wait only for every case, not for its scores.

    Returns:
        The item runs Langfuse shows by case id, and a description of each case and
        score it still lacks, empty when it shows everything.

    Raises:
        PushMismatchError: Langfuse shows a case or score that differs from the run, or
            a case in a dataset or item it does not map to, or on another input.
    """
    by_case = _results_by_case(run)
    expected_digests = {case_id: _case_digest(results) for case_id, results in by_case.items()}
    expected_scores = {
        case_id: {e.name: float(e.value) for e in case_scores(results)}
        for case_id, results in by_case.items()
    }

    deadline = time.monotonic() + timeout_s
    while True:
        held = _read_pushed(
            client=client,
            run=run,
            run_name=run_name,
            groups=groups,
            project_id=project_id,
            input_digests=input_digests,
        )
        _check_matches(
            held=held, digests=expected_digests, scores=expected_scores, run_name=run_name
        )
        missing = _missing(
            held=held,
            scores={case_id: {} for case_id in by_case} if cases_only else expected_scores,
        )
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
    experiment_id: str,
    data: list[DatasetItem],
    outputs: Mapping[str, object],
) -> None:
    """Send one item run per item of one dataset through ``run_experiment``.

    It runs no evaluators: ``_write_scores`` writes every score once the item runs show.
    """

    def task(*, item: ExperimentItem, **_: object) -> object:
        """Return the output the run holds for the item's case."""
        if not isinstance(item, DatasetItem):
            raise TypeError(f"expected a DatasetItem, got {type(item).__name__}")
        return outputs[_case_id_of(item.metadata)]

    manifest = run.manifest
    result = client.run_experiment(
        name=manifest.suite,
        run_name=run_name,
        experiment_id=experiment_id,
        description=f"{manifest.suite} run {manifest.run_id} at {manifest.revision}",
        data=data,
        task=task,
        evaluators=[],
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


def _read_pushed(
    *,
    client: Langfuse,
    run: Run,
    run_name: str,
    groups: Sequence[_DatasetCases],
    project_id: str,
    input_digests: Mapping[str, str],
) -> dict[str, _ItemRun]:
    """Read the item runs Langfuse shows under an experiment in each dataset, by case id.

    Raises:
        PushMismatchError: An item run lacks the metadata a push writes, is of a case the
            run does not map to its dataset, is of another item than the case maps to,
            or holds another input than the run evaluated.
    """
    found: dict[str, _ItemRun] = {}
    for group in groups:
        shown = _read_experiment_items(
            client=client,
            run=run,
            run_name=run_name,
            experiment_id=_experiment_id(
                project_id=project_id, dataset_id=group.dataset_id, run_name=run_name
            ),
            group=group,
        )
        if stray := sorted(set(shown) - set(group.items)):
            raise PushMismatchError(
                f"Langfuse shows cases {stray} under {run_name} in dataset "
                f"{group.dataset!r}, which the run does not map there"
            )
        for case_id, item_run in shown.items():
            if item_run.item_id != (mapped := group.items[case_id].id):
                raise PushMismatchError(
                    f"case {case_id!r} under {run_name} in dataset {group.dataset!r} is an "
                    f"item run of {item_run.item_id!r}, not of {mapped!r}"
                )
            if not _read_back_input_matches(item_run.input, input_digests[case_id]):
                raise PushMismatchError(
                    f"case {case_id!r} under {run_name} in dataset {group.dataset!r} ran on "
                    "another input than the run evaluated"
                )
        found |= shown
    return found


def _read_experiment_items(
    *, client: Langfuse, run: Run, run_name: str, experiment_id: str, group: _DatasetCases
) -> dict[str, _ItemRun]:
    """Read the item runs Langfuse shows under an experiment in one dataset, newest first.

    The read filters on the experiment id a push writes under as well as its name, so
    item runs another experiment of the same name holds stay out of the read-back.

    Raises:
        PushMismatchError: An item run lacks the metadata a push writes.
    """
    found: dict[str, _ItemRun] = {}
    cursor: str | None = None
    while True:
        page = client.api.experiments.list_items(
            # No item run of this run's push can start before the run did.
            from_start_time=run.manifest.started_at,
            experiment_id=experiment_id,
            experiment_name=run_name,
            dataset_id=group.dataset_id,
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
                    observation_id=item.id,
                    start_time=item.start_time,
                    item_id=item.experiment_item_id,
                    input=item.input,
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
            break

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
        PushMismatchError: A case's records or output differ, or a score has another value.
    """
    for case_id, item_run in held.items():
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


def _changed_fields(*, held: DatasetItem, wanted: Mapping[str, object]) -> list[str]:
    """The fields of a held item that a write of ``wanted`` would change.

    A ``None`` field keeps the stored value, as Langfuse treats it, so it never changes.
    """
    return [
        name
        for name, value in wanted.items()
        if value is not None and _digest(value) != _digest(getattr(held, name))
    ]


def _held_dataset(*, client: Langfuse, dataset: str, create: bool) -> _HeldDataset | None:
    """Return a dataset's id and items by id, or None when it is missing.

    A missing dataset is created first when ``create`` is set, and then read as empty.
    """
    try:
        held = client.get_dataset(dataset)
    except NotFoundError:
        if create:
            client.create_dataset(name=dataset)
            logger.info("created dataset %s", dataset)
        return None
    return _HeldDataset(dataset_id=held.id, items={item.id: item for item in held.items})


def _group_cases(
    *, client: Langfuse, case_ids: Sequence[str], item_of: Callable[[str], DatasetItemRef]
) -> list[_DatasetCases]:
    """Group cases by the dataset ``item_of`` maps them to, each with its held item.

    Raises:
        ValueError: Two cases map to one item.
        UnknownCaseError: A case has no item in the dataset it maps to.
    """
    refs = {case_id: item_of(case_id) for case_id in case_ids}
    if len(set(refs.values())) != len(refs):
        raise ValueError(f"cases map to the same dataset item: {refs}")
    by_dataset: defaultdict[str, dict[str, str]] = defaultdict(dict)
    for case_id, ref in refs.items():
        by_dataset[ref.dataset][case_id] = ref.item_id

    groups = []
    unknown = []
    for dataset, item_ids in sorted(by_dataset.items()):
        held = _held_dataset(client=client, dataset=dataset, create=False)
        lacking = [
            case_id
            for case_id, item_id in item_ids.items()
            if held is None or item_id not in held.items
        ]
        if held is None or lacking:
            unknown.append(f"dataset {dataset!r} holds no item for cases {lacking}")
            continue
        groups.append(
            _DatasetCases(
                dataset=dataset,
                dataset_id=held.dataset_id,
                items={case_id: held.items[item_id] for case_id, item_id in item_ids.items()},
            )
        )
    if unknown:
        raise UnknownCaseError("; ".join(unknown))
    return groups


def _case_input_digests(by_case: Mapping[str, Sequence[CaseResult]]) -> dict[str, str]:
    """The stored input digest of each case, which all its trials share.

    Raises:
        ValueError: A case's trials hold different input digests.
    """
    digests = {}
    for case_id, results in by_case.items():
        found = {result.input_digest for result in results}
        if len(found) != 1:
            raise ValueError(f"case {case_id!r}: trials hold different input digests {found}")
        digests[case_id] = found.pop()
    return digests


def _check_inputs(*, groups: Sequence[_DatasetCases], input_digests: Mapping[str, str]) -> None:
    """Refuse any case whose dataset item input is not the input the run evaluated.

    Raises:
        InputMismatchError: Some item inputs differ, named by dataset and case.
    """
    differ = [
        f"dataset {group.dataset!r} cases {cases}"
        for group in groups
        if (cases := _cases_on_other_input(group=group, input_digests=input_digests))
    ]
    if differ:
        raise InputMismatchError(
            f"item inputs differ from the inputs the run evaluated: {'; '.join(differ)}"
        )


def _cases_on_other_input(*, group: _DatasetCases, input_digests: Mapping[str, str]) -> list[str]:
    """The cases of one dataset whose item input is not the input the run evaluated."""
    return [
        case_id
        for case_id, item in group.items.items()
        if not _input_matches(item.input, input_digests[case_id])
    ]


def _input_matches(value: object, digest: str) -> bool:
    """Whether a value read from Langfuse is the input a stored input digest names."""
    return input_digest(value) == digest


def _read_back_input_matches(value: object, digest: str) -> bool:
    """Whether an item run's input, as read back, is the input a digest names.

    Langfuse may return an item run's input as JSON text, or a string input as the
    string itself, so a string matches either as it is or parsed as JSON. Parsing alone
    would read the string input ``"2024"`` as the number 2024.
    """
    if _input_matches(value, digest):
        return True
    return isinstance(value, str) and _input_matches(_parsed(value), digest)


def _write_scores(
    *,
    client: Langfuse,
    run_name: str,
    item_runs: Mapping[str, _ItemRun],
    dataset_of: Mapping[str, str],
    scores: Mapping[str, list[Evaluation]],
) -> int:
    """Write each score an item run does not show yet, and return how many were written.

    Every score of a push goes through here, under an id fixed by the run name, dataset,
    item and check and with the item run's start time as its timestamp, so writing one
    again overwrites it instead of adding a second.
    """
    written = 0
    for case_id, item_run in item_runs.items():
        for evaluation in scores[case_id]:
            if evaluation.name in item_run.scores:
                continue
            client.create_score(
                trace_id=item_run.trace_id,
                # The observation Langfuse reads an item run's scores back by.
                observation_id=item_run.observation_id,
                name=evaluation.name,
                value=float(evaluation.value),
                data_type=_SCORE_DATA_TYPE,
                score_id=_score_id(
                    run_name=run_name,
                    dataset=dataset_of[case_id],
                    item_id=item_run.item_id,
                    check=evaluation.name,
                ),
                # Langfuse reads an item run's scores back only within a day of its
                # items, and a resent score overwrites one of the same id and timestamp.
                timestamp=item_run.start_time,
            )
            written += 1
    return written


def _score_id(*, run_name: str, dataset: str, item_id: str, check: str) -> str:
    """The id of a score, the same each time it is written."""
    return str(uuid.uuid5(_SCORE_ID_NAMESPACE, json.dumps([run_name, dataset, item_id, check])))


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
    return input_digest(value)


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
