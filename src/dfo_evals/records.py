"""Run records: the on-disk format every eval suite writes and every report reads.

A run lives in one directory, ``<root>/<suite>/<run_id>/``, holding ``manifest.json``
(one ``RunManifest``) and ``results.jsonl`` (one ``CaseResult`` per line).

``iter_runs`` reads a directory holding ``manifest.json`` as a complete run, so an
uploader that copies run files one at a time writes ``results.jsonl`` first and
``manifest.json`` last. ``write_run`` is atomic on a local disk only.

A suite cannot be named ``site``: the results bucket keeps the page at
``site/index.html``, beside the suite directories.

Writers record ``SCHEMA_VERSION`` in every manifest. Readers refuse a field they do not
know, so a record format that gains a field, even an optional one, takes a new
``SCHEMA_VERSION``, and the models of the release that adds it must still validate every
earlier version, since the bucket keeps every run. ``read_run`` accepts a manifest whose
``schema_version`` is an integer from 1 to ``SCHEMA_VERSION`` and validates it against
the current models. It refuses a newer version, or a value that is not an integer of 1
or more, with ``RecordSchemaError``, and the results page lists that run as unreadable
until the publisher's dfo-evals catches up.
"""

import json
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

SCHEMA_VERSION: Final = 1
MANIFEST_FILE: Final = "manifest.json"
RESULTS_FILE: Final = "results.jsonl"
# The page lives at <root>/site/index.html, so no suite may take that directory.
RESERVED_SUITE: Final = "site"
# Name of the single FAIL check recorded for a trial whose task raised.
TASK_CHECK_NAME: Final = "task"
# A revision holding this cannot name a Langfuse experiment, which reads it as a list
# separator, so write_run refuses it.
REVISION_SEPARATOR: Final = ","
# The first schema version; readers accept every version from this to SCHEMA_VERSION.
FIRST_SCHEMA_VERSION: Final = 1

# Run and suite names become directory names, so they must not escape or hide.
_SAFE_NAME_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
# Temp directories start with "." so a safe run_id can never collide with one.
_TEMP_PREFIX: Final = "."
_TEMP_SUFFIX: Final = ".tmp"
# mkdtemp creates 0o700; runs are read by other users (CI, report builds).
_RUN_DIR_MODE: Final = 0o755

NonEmptyStr = Annotated[str, Field(min_length=1)]
SafeName = Annotated[str, Field(pattern=_SAFE_NAME_PATTERN)]

# Non-finite floats would serialise as null and read back changed, so reject them.
_MODEL_CONFIG: Final = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class RecordSchemaError(ValueError):
    """A run record on disk does not match the schema this library reads."""


class Verdict(StrEnum):
    """Outcome of one check on one case."""

    PASS = "pass"
    FAIL = "fail"
    UNVERIFIABLE = "unverifiable"
    UNREACHED = "unreached"
    NOT_APPLICABLE = "n/a"


class FaultClass(StrEnum):
    """Who is at fault for a failed or unscored result.

    ``UPSTREAM`` means a suspected defect in the source data rather than in the agent.
    """

    AGENT = "agent"
    UPSTREAM = "upstream"
    REFERENCE = "reference"
    GRADER = "grader"
    INFRA = "infra"


class CheckResult(BaseModel):
    """Verdict of one named check on one case."""

    model_config = _MODEL_CONFIG

    name: NonEmptyStr
    verdict: Verdict
    detail: str | None = None


class CaseResult(BaseModel):
    """Result of one trial of one case, with a verdict per check.

    Attributes:
        duration_s: Time the task took in seconds, or None when unknown, such as
            for a task that raised.
        cost_usd: Cost of the trial in nominal USD, or None when unknown.
        tags: Grouping labels such as corpus file or family, used by reports. Tags
            belong to the case, so every trial of a case carries the same set.
    """

    model_config = _MODEL_CONFIG

    case_id: NonEmptyStr
    trial: int = Field(ge=1)
    input_digest: str
    checks: tuple[CheckResult, ...]
    fault: FaultClass | None = None
    duration_s: float | None = Field(ge=0)
    cost_usd: float | None = Field(ge=0)
    trace_id: str | None = None
    tags: tuple[str, ...] = ()

    @field_validator("checks")
    @classmethod
    def _check_names_unique(cls, checks: tuple[CheckResult, ...]) -> tuple[CheckResult, ...]:
        """Reject a case result that names the same check twice."""
        duplicates = _duplicates(check.name for check in checks)
        if duplicates:
            raise ValueError(f"duplicate check names: {duplicates}")
        return checks

    @field_validator("checks")
    @classmethod
    def _task_check_alone(cls, checks: tuple[CheckResult, ...]) -> tuple[CheckResult, ...]:
        """Reject a ``task`` check beside other checks.

        The name marks a task that raised, whose trial holds that one check, and scoring
        fails any trial that carries it.
        """
        if len(checks) > 1 and any(check.name == TASK_CHECK_NAME for check in checks):
            raise ValueError(
                f"check name {TASK_CHECK_NAME!r} is reserved for a task that raised and "
                "must be the trial's only check"
            )
        return checks


class RunManifest(BaseModel):
    """Metadata describing one run of one suite.

    Attributes:
        environment: Settings the run depended on, such as app URL and model.
        headline_check: Name of the check that decides whether a trial passes, or None
            for a trial to pass when no check fails and at least one passes. Stored with
            the run so every later report scores it by the rule it ran under.
    """

    model_config = _MODEL_CONFIG

    schema_version: int = Field(default=SCHEMA_VERSION, ge=FIRST_SCHEMA_VERSION, strict=True)
    suite: SafeName
    run_id: SafeName
    started_at: AwareDatetime
    finished_at: AwareDatetime | None
    revision: str
    environment: dict[str, str]
    dataset_digest: str
    headline_check: NonEmptyStr | None = None

    @field_validator("schema_version")
    @classmethod
    def _schema_version_not_newer(cls, version: int) -> int:
        """Reject a schema version newer than this library reads."""
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {version} is newer than this dfo-evals reads "
                f"(up to {SCHEMA_VERSION})"
            )
        return version

    @field_validator("headline_check")
    @classmethod
    def _headline_is_not_the_task_check(cls, headline_check: str | None) -> str | None:
        """Reject the name reserved for a task that raised."""
        if headline_check == TASK_CHECK_NAME:
            raise ValueError(
                f"headline_check {TASK_CHECK_NAME!r} is reserved for a task that raised"
            )
        return headline_check

    @field_validator("suite")
    @classmethod
    def _suite_not_reserved(cls, suite: str) -> str:
        """Reject the suite name the results page's directory takes."""
        if suite == RESERVED_SUITE:
            raise ValueError(f"suite name {RESERVED_SUITE!r} is reserved for the results page")
        return suite

    @model_validator(mode="after")
    def _finished_not_before_started(self) -> Self:
        """Reject a run that finished before it started."""
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError(
                f"finished_at {self.finished_at.isoformat()} is before "
                f"started_at {self.started_at.isoformat()}"
            )
        return self


class Run(BaseModel):
    """A manifest and every case result it produced."""

    model_config = _MODEL_CONFIG

    manifest: RunManifest
    results: tuple[CaseResult, ...]

    @field_validator("results")
    @classmethod
    def _case_trials_unique(cls, results: tuple[CaseResult, ...]) -> tuple[CaseResult, ...]:
        """Reject a run holding the same trial of the same case twice."""
        duplicates = _duplicates((result.case_id, result.trial) for result in results)
        if duplicates:
            raise ValueError(f"duplicate (case_id, trial) pairs: {duplicates}")
        return results

    @field_validator("results")
    @classmethod
    def _tags_per_case(cls, results: tuple[CaseResult, ...]) -> tuple[CaseResult, ...]:
        """Reject a run whose trials of one case carry different tag sets.

        Reports group cases by tag and count every trial of each member case, so a tag
        held by only some trials of a case would leave the group without them.
        """
        tags_by_case: dict[str, frozenset[str]] = {}
        for result in results:
            tags = frozenset(result.tags)
            first = tags_by_case.setdefault(result.case_id, tags)
            if tags != first:
                raise ValueError(
                    f"case {result.case_id!r} trial {result.trial} has tags {sorted(tags)}, "
                    f"another trial has {sorted(first)}; tags must be the same on every trial"
                )
        return results

    @field_validator("results")
    @classmethod
    def _input_digest_per_case(cls, results: tuple[CaseResult, ...]) -> tuple[CaseResult, ...]:
        """Reject a run whose trials of one case carry different input digests.

        Every trial of a case runs on the case's one input, and a Langfuse push maps the
        case to one dataset item holding that input.
        """
        digest_by_case: dict[str, str] = {}
        for result in results:
            first = digest_by_case.setdefault(result.case_id, result.input_digest)
            if result.input_digest != first:
                raise ValueError(
                    f"case {result.case_id!r} trial {result.trial} has input digest "
                    f"{result.input_digest!r}, another trial has {first!r}; every trial "
                    "of a case must have the same input digest"
                )
        return results

    @field_validator("results")
    @classmethod
    def _same_trial_count_per_case(cls, results: tuple[CaseResult, ...]) -> tuple[CaseResult, ...]:
        """Reject a run whose cases hold different numbers of trials.

        Summaries divide by one trial count for the whole run, so an interrupted or
        merged run with uneven cases would save and then fail every report.
        """
        counts = Counter(result.case_id for result in results)
        if len(set(counts.values())) > 1:
            raise ValueError(
                f"cases hold different numbers of trials: {dict(sorted(counts.items()))}"
            )
        return results

    @model_validator(mode="after")
    def _scored_trials_hold_the_headline(self) -> Self:
        """Reject a trial the headline check would decide that lacks that check.

        A trial with a fault, or whose task raised, is scored without its headline
        check, so only the others must hold it.
        """
        headline = self.manifest.headline_check
        if headline is None:
            return self
        for result in self.results:
            if result.fault is not None or task_raised(result):
                continue
            if all(check.name != headline for check in result.checks):
                raise ValueError(
                    f"suite {self.manifest.suite!r} run {self.manifest.run_id!r}: case "
                    f"{result.case_id!r} trial {result.trial} has no headline check {headline!r}"
                )
        return self


def task_raised(result: CaseResult) -> bool:
    """Whether the trial's task raised, recorded as a FAIL ``task`` check."""
    return any(
        check.name == TASK_CHECK_NAME and check.verdict is Verdict.FAIL for check in result.checks
    )


def write_run(*, root: Path, run: Run) -> Path:
    """Write a run to ``<root>/<suite>/<run_id>/`` atomically.

    Both files go into a temporary sibling directory that is renamed into place, so a
    crashed write leaves no partial run.

    Args:
        root: Directory holding one subdirectory per suite.
        run: The run to write.

    Returns:
        The run directory.

    Raises:
        ValueError: The revision holds a comma, which Langfuse reads as a list separator
            in experiment names. Nothing is written.
        FileExistsError: The run directory already exists.
    """
    if REVISION_SEPARATOR in run.manifest.revision:
        raise ValueError(f"revision {run.manifest.revision!r} holds a comma")
    suite_dir = root / run.manifest.suite
    run_dir = suite_dir / run.manifest.run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    suite_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{_TEMP_PREFIX}{run.manifest.run_id}.", suffix=_TEMP_SUFFIX, dir=suite_dir
        )
    )
    try:
        temp_dir.chmod(_RUN_DIR_MODE)
        _write_text(temp_dir / MANIFEST_FILE, run.manifest.model_dump_json(indent=2) + "\n")
        _write_text(
            temp_dir / RESULTS_FILE,
            "".join(result.model_dump_json() + "\n" for result in run.results),
        )
        # rename refuses a non-empty target, closing the race after the exists() check.
        try:
            temp_dir.rename(run_dir)
        except OSError as error:
            if run_dir.exists():
                raise FileExistsError(f"run directory already exists: {run_dir}") from error
            raise
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return run_dir


def read_run(directory: Path) -> Run:
    """Read and validate the run stored in a run directory.

    Args:
        directory: A run directory written by ``write_run``.

    Returns:
        The validated run.

    Raises:
        FileNotFoundError: ``manifest.json`` or ``results.jsonl`` is missing.
        RecordSchemaError: The manifest has another schema version, or a file fails
            validation.
    """
    manifest = _read_manifest(directory / MANIFEST_FILE)
    results = _read_results(directory / RESULTS_FILE)
    try:
        return Run(manifest=manifest, results=results)
    except ValidationError as error:
        raise RecordSchemaError(f"{directory}: invalid run: {error}") from error


def iter_runs(*, root: Path, suite: str | None = None) -> Iterator[Path]:
    """Yield run directories under a root, sorted by suite then run_id.

    Args:
        root: Directory holding one subdirectory per suite.
        suite: Only yield runs of this suite. None yields every suite.

    Yields:
        Each directory that holds a ``manifest.json``, skipping temporary directories.

    Raises:
        ValueError: ``suite`` is not a safe directory name.
        FileNotFoundError: ``root``, or the named suite under it, does not exist.
    """
    if suite is not None and not re.fullmatch(_SAFE_NAME_PATTERN, suite):
        raise ValueError(f"suite {suite!r} is not a safe directory name")
    if not root.is_dir():
        raise FileNotFoundError(f"run root not found: {root}")
    if suite is not None and not (root / suite).is_dir():
        raise FileNotFoundError(f"suite not found under {root}: {suite}")
    suite_dirs = [root / suite] if suite is not None else sorted(root.iterdir())
    for suite_dir in suite_dirs:
        if not suite_dir.is_dir():
            continue
        for run_dir in sorted(suite_dir.iterdir()):
            if run_dir.name.startswith(_TEMP_PREFIX):
                continue
            if (run_dir / MANIFEST_FILE).is_file():
                yield run_dir


def _read_manifest(path: Path) -> RunManifest:
    """Read a manifest, checking its schema version before validating the rest."""
    text = _read_text(path)
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise RecordSchemaError(f"{path}: not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise RecordSchemaError(f"{path}: not a JSON object")
    if "schema_version" not in raw:
        raise RecordSchemaError(f"{path}: schema_version is missing")
    version = raw["schema_version"]
    # bool is an int subclass, but true is no version.
    if not isinstance(version, int) or isinstance(version, bool) or version < FIRST_SCHEMA_VERSION:
        raise RecordSchemaError(
            f"{path}: schema_version {version!r} is not an integer from {FIRST_SCHEMA_VERSION}"
        )
    if version > SCHEMA_VERSION:
        raise RecordSchemaError(
            f"{path}: schema_version {version} is newer than this dfo-evals reads "
            f"(up to {SCHEMA_VERSION})"
        )
    try:
        return RunManifest.model_validate(raw)
    except ValidationError as error:
        raise RecordSchemaError(f"{path}: invalid manifest: {error}") from error


def _read_results(path: Path) -> list[CaseResult]:
    """Read one CaseResult per line; empty lines are errors."""
    # Split on "\n" only: str.splitlines also breaks on U+2028 and other separators
    # that JSON strings may hold unescaped.
    lines = _read_text(path).split("\n")
    if lines[-1] == "":
        lines.pop()
    results: list[CaseResult] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise RecordSchemaError(f"{path}:{line_number}: empty line")
        try:
            results.append(CaseResult.model_validate_json(line))
        except ValidationError as error:
            raise RecordSchemaError(
                f"{path}:{line_number}: invalid case result: {error}"
            ) from error
    return results


def _read_text(path: Path) -> str:
    """Read a UTF-8 file, naming the path when it is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"run record file not found: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise RecordSchemaError(f"{path}: not valid UTF-8 at byte {error.start}") from error


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text with ``\\n`` line endings on every platform."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _duplicates[T](items: Iterable[T]) -> list[T]:
    """Return the items that occur more than once, in first-seen order."""
    return [item for item, count in Counter(items).items() if count > 1]
