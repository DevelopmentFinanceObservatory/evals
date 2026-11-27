"""Tests for run records and their files."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from dfo_evals import records
from dfo_evals.records import (
    SCHEMA_VERSION,
    TASK_CHECK_NAME,
    CaseResult,
    CheckResult,
    FaultClass,
    RecordSchemaError,
    Run,
    RunManifest,
    Verdict,
    iter_runs,
    read_run,
    write_run,
)

STARTED_AT = datetime(2026, 11, 24, 9, 0, tzinfo=UTC)


def make_manifest(*, suite: str = "apis-search", run_id: str = "run-1", **overrides) -> RunManifest:
    fields = {
        "suite": suite,
        "run_id": run_id,
        "started_at": STARTED_AT,
        "finished_at": STARTED_AT + timedelta(minutes=5),
        "revision": "abc123",
        "environment": {"app_url": "http://localhost:8000", "model": "test-model"},
        "dataset_digest": "sha256:dataset",
    } | overrides
    return RunManifest(**fields)


def make_case(*, case_id: str = "case-1", trial: int = 1, **overrides) -> CaseResult:
    fields = {
        "case_id": case_id,
        "trial": trial,
        "input_digest": "sha256:input",
        "checks": (
            CheckResult(name="country_match", verdict=Verdict.PASS),
            CheckResult(name="year_match", verdict=Verdict.FAIL, detail="got 2021"),
            CheckResult(name="unit_match", verdict=Verdict.NOT_APPLICABLE),
        ),
        "fault": FaultClass.AGENT,
        "duration_s": 1.5,
        "cost_usd": 0.002,
        "trace_id": "trace-1",
        "tags": ("corpus/oda.yaml", "family:share"),
    } | overrides
    return CaseResult(**fields)


def make_run(*, suite: str = "apis-search", run_id: str = "run-1") -> Run:
    return Run(
        manifest=make_manifest(suite=suite, run_id=run_id),
        results=(make_case(case_id="case-1"), make_case(case_id="case-2", cost_usd=None)),
    )


def test_a_run_written_then_read_comes_back_equal(tmp_path: Path) -> None:
    run = make_run()

    run_dir = write_run(root=tmp_path, run=run)

    assert run_dir == tmp_path / "apis-search" / "run-1"
    assert read_run(run_dir) == run


def test_a_detail_holding_unicode_line_separators_round_trips(tmp_path: Path) -> None:
    detail = "got\u2028 2021\u0085 and\x1c 2022"
    checks = (CheckResult(name="year_match", verdict=Verdict.FAIL, detail=detail),)
    run = Run(manifest=make_manifest(), results=(make_case(checks=checks),))

    run_dir = write_run(root=tmp_path, run=run)

    assert read_run(run_dir) == run


def test_written_files_are_utf8_with_one_result_per_line(tmp_path: Path) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())

    raw = (run_dir / "results.jsonl").read_bytes()

    assert b"\r" not in raw
    assert len(raw.decode("utf-8").splitlines()) == 2


def test_a_run_with_no_results_round_trips(tmp_path: Path) -> None:
    run = Run(manifest=make_manifest(), results=())

    assert read_run(write_run(root=tmp_path, run=run)) == run


def test_writing_over_an_existing_run_is_refused(tmp_path: Path) -> None:
    write_run(root=tmp_path, run=make_run())

    with pytest.raises(FileExistsError, match="run-1"):
        write_run(root=tmp_path, run=make_run())


def test_a_write_leaves_no_temp_directory_behind(tmp_path: Path) -> None:
    write_run(root=tmp_path, run=make_run())

    assert [path.name for path in (tmp_path / "apis-search").iterdir()] == ["run-1"]


def test_duplicate_check_names_are_rejected() -> None:
    checks = (
        CheckResult(name="country_match", verdict=Verdict.PASS),
        CheckResult(name="country_match", verdict=Verdict.FAIL),
    )

    with pytest.raises(ValidationError, match="duplicate check names"):
        make_case(checks=checks)


def test_an_empty_check_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CheckResult(name="", verdict=Verdict.PASS)


def test_a_task_check_beside_other_checks_is_rejected() -> None:
    with pytest.raises(ValidationError, match="reserved for a task that raised"):
        make_case(
            checks=(
                CheckResult(name="answer", verdict=Verdict.PASS),
                CheckResult(name="task", verdict=Verdict.FAIL),
            )
        )


def test_duplicate_case_and_trial_pairs_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        Run(manifest=make_manifest(), results=(make_case(), make_case()))


def test_trials_of_one_case_with_different_tags_are_rejected() -> None:
    with pytest.raises(ValidationError, match="case 'case-1' trial 2 has tags"):
        Run(
            manifest=make_manifest(),
            results=(make_case(trial=1, tags=("dac",)), make_case(trial=2, tags=())),
        )


def test_cases_with_different_numbers_of_trials_are_rejected() -> None:
    with pytest.raises(ValidationError, match="different numbers of trials"):
        Run(
            manifest=make_manifest(),
            results=(
                make_case(case_id="a", trial=1),
                make_case(case_id="a", trial=2),
                make_case(case_id="b", trial=1),
            ),
        )


def test_a_stored_manifest_whose_revision_holds_a_comma_reads_back(tmp_path: Path) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "abc,dirty"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert read_run(run_dir).manifest.revision == "abc,dirty"


def test_writing_a_run_whose_revision_holds_a_comma_is_refused(tmp_path: Path) -> None:
    run = Run(manifest=make_manifest(revision="abc,dirty"), results=(make_case(),))

    with pytest.raises(ValueError, match="holds a comma"):
        write_run(root=tmp_path, run=run)

    assert list(tmp_path.iterdir()) == []


def test_trials_of_one_case_with_different_input_digests_are_rejected() -> None:
    with pytest.raises(ValidationError, match=r"case 'case-1' trial 2 has input digest"):
        Run(
            manifest=make_manifest(),
            results=(
                make_case(trial=1),
                make_case(trial=2, input_digest="sha256:other"),
            ),
        )


def test_different_cases_may_have_different_input_digests() -> None:
    run = Run(
        manifest=make_manifest(),
        results=(make_case(case_id="a"), make_case(case_id="b", input_digest="sha256:other")),
    )

    assert len(run.results) == 2


def test_trials_of_one_case_may_list_the_same_tags_in_any_order() -> None:
    run = Run(
        manifest=make_manifest(),
        results=(
            make_case(trial=1, tags=("dac", "family:share")),
            make_case(trial=2, tags=("family:share", "dac")),
        ),
    )

    assert len(run.results) == 2


def test_the_task_check_cannot_be_the_headline() -> None:
    with pytest.raises(ValidationError, match="reserved for a task that raised"):
        make_manifest(headline_check=TASK_CHECK_NAME)


def test_a_trial_lacking_the_headline_check_is_rejected_naming_suite_run_and_case() -> None:
    case = make_case(case_id="case-7", fault=None)

    with pytest.raises(
        ValidationError, match=r"'apis-search' run 'run-1': case 'case-7' trial 1 .*'answer'"
    ):
        Run(manifest=make_manifest(headline_check="answer"), results=(case,))


def test_faulted_and_crashed_trials_need_no_headline_check() -> None:
    crashed = make_case(
        case_id="crashed",
        fault=None,
        checks=(CheckResult(name=TASK_CHECK_NAME, verdict=Verdict.FAIL, detail="boom"),),
    )
    faulted = make_case(case_id="faulted", fault=FaultClass.INFRA)

    run = Run(manifest=make_manifest(headline_check="answer"), results=(crashed, faulted))

    assert len(run.results) == 2


def test_a_headline_check_round_trips(tmp_path: Path) -> None:
    case = make_case(fault=None)
    run = Run(manifest=make_manifest(headline_check="year_match"), results=(case,))

    assert read_run(write_run(root=tmp_path, run=run)) == run


def test_the_same_case_in_different_trials_is_accepted() -> None:
    run = Run(manifest=make_manifest(), results=(make_case(trial=1), make_case(trial=2)))

    assert len(run.results) == 2


def test_the_suite_name_the_results_page_takes_is_rejected() -> None:
    with pytest.raises(ValidationError, match="reserved"):
        make_manifest(suite="site")


@pytest.mark.parametrize("run_id", ["", ".hidden", "../escape", "a/b", "has space", "-lead"])
def test_a_run_id_unsafe_as_a_directory_name_is_rejected(run_id: str) -> None:
    with pytest.raises(ValidationError):
        make_manifest(run_id=run_id)


def test_a_run_finished_before_it_started_is_rejected() -> None:
    with pytest.raises(ValidationError, match="before"):
        make_manifest(finished_at=STARTED_AT - timedelta(seconds=1))


def test_a_naive_start_time_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_manifest(started_at=datetime(2026, 11, 24, 9, 0), finished_at=None)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CheckResult(name="x", verdict=Verdict.PASS, extra="nope")


def _with_schema_version(run_dir: Path, version: object) -> None:
    """Rewrite a stored manifest's schema_version."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_a_manifest_from_a_newer_schema_version_is_refused_as_newer(tmp_path: Path) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())
    _with_schema_version(run_dir, SCHEMA_VERSION + 1)

    with pytest.raises(RecordSchemaError, match="newer than this dfo-evals reads") as caught:
        read_run(run_dir)

    message = str(caught.value)
    assert str(SCHEMA_VERSION + 1) in message
    assert str(SCHEMA_VERSION) in message


@pytest.mark.parametrize("version", range(1, SCHEMA_VERSION + 1))
def test_a_manifest_of_every_supported_schema_version_reads(tmp_path: Path, version: int) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())
    _with_schema_version(run_dir, version)

    assert read_run(run_dir).manifest.schema_version == version


def test_a_reader_of_a_later_schema_version_reads_earlier_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())
    later = SCHEMA_VERSION + 1
    monkeypatch.setattr(records, "SCHEMA_VERSION", later)

    assert read_run(run_dir).manifest.schema_version == SCHEMA_VERSION
    _with_schema_version(run_dir, later)
    assert read_run(run_dir).manifest.schema_version == later
    _with_schema_version(run_dir, later + 1)
    with pytest.raises(RecordSchemaError, match="newer than this dfo-evals reads"):
        read_run(run_dir)


@pytest.mark.parametrize("version", [0, -1, "1", 1.0, True, None])
def test_a_manifest_whose_schema_version_is_no_valid_integer_is_refused(
    tmp_path: Path, version: object
) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())
    _with_schema_version(run_dir, version)

    with pytest.raises(RecordSchemaError, match="is not an integer from 1"):
        read_run(run_dir)


def test_a_manifest_without_a_schema_version_is_refused_as_missing(tmp_path: Path) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["schema_version"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RecordSchemaError, match="schema_version is missing"):
        read_run(run_dir)


def test_a_manifest_that_is_not_a_json_object_is_refused(tmp_path: Path) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())
    (run_dir / "manifest.json").write_text("[1]", encoding="utf-8")

    with pytest.raises(RecordSchemaError, match="not a JSON object"):
        read_run(run_dir)


def test_a_written_manifest_holds_the_current_schema_version(tmp_path: Path) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == SCHEMA_VERSION


def test_a_corrupt_results_line_reports_its_line_number(tmp_path: Path) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())
    results_path = run_dir / "results.jsonl"
    lines = results_path.read_text(encoding="utf-8").splitlines()
    lines[1] = '{"case_id": "broken"}'
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RecordSchemaError, match=r"results\.jsonl:2:") as caught:
        read_run(run_dir)

    assert isinstance(caught.value.__cause__, ValidationError)


def test_an_empty_results_line_is_an_error(tmp_path: Path) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())
    results_path = run_dir / "results.jsonl"
    lines = results_path.read_text(encoding="utf-8").splitlines()
    results_path.write_text(f"{lines[0]}\n\n{lines[1]}\n", encoding="utf-8")

    with pytest.raises(RecordSchemaError, match=r"results\.jsonl:2: empty line"):
        read_run(run_dir)


def test_a_missing_results_file_names_the_path(tmp_path: Path) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())
    (run_dir / "results.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match=r"results\.jsonl"):
        read_run(run_dir)


def test_iter_runs_yields_runs_by_suite_then_run_id_and_skips_temp_dirs(tmp_path: Path) -> None:
    for suite, run_id in [("b-suite", "run-2"), ("a-suite", "run-9"), ("b-suite", "run-1")]:
        write_run(root=tmp_path, run=make_run(suite=suite, run_id=run_id))
    temp_dir = tmp_path / "a-suite" / ".run-0.abc.tmp"
    temp_dir.mkdir()
    (temp_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "a-suite" / "no-manifest").mkdir()

    found = [(path.parent.name, path.name) for path in iter_runs(root=tmp_path)]

    assert found == [("a-suite", "run-9"), ("b-suite", "run-1"), ("b-suite", "run-2")]


def test_iter_runs_filters_by_suite(tmp_path: Path) -> None:
    write_run(root=tmp_path, run=make_run(suite="a-suite", run_id="run-1"))
    write_run(root=tmp_path, run=make_run(suite="b-suite", run_id="run-1"))

    found = list(iter_runs(root=tmp_path, suite="b-suite"))

    assert found == [tmp_path / "b-suite" / "run-1"]


def test_iter_runs_on_a_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="absent"):
        list(iter_runs(root=tmp_path / "absent"))


def test_iter_runs_on_a_missing_suite_raises(tmp_path: Path) -> None:
    write_run(root=tmp_path, run=make_run(suite="a-suite"))

    with pytest.raises(FileNotFoundError, match="b-suite"):
        list(iter_runs(root=tmp_path, suite="b-suite"))


def test_iter_runs_rejects_a_suite_that_escapes_the_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="safe directory name"):
        list(iter_runs(root=tmp_path, suite="../outside"))


@pytest.mark.parametrize("version", [0, SCHEMA_VERSION + 1, True, "1"])
def test_a_manifest_outside_the_supported_schema_versions_cannot_be_built(
    version: object,
) -> None:
    with pytest.raises(ValidationError):
        make_manifest(schema_version=version)


def test_an_unknown_duration_round_trips(tmp_path: Path) -> None:
    run = Run(manifest=make_manifest(), results=(make_case(duration_s=None),))

    assert read_run(write_run(root=tmp_path, run=run)) == run


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_non_finite_durations_and_costs_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        make_case(duration_s=value)
    with pytest.raises(ValidationError):
        make_case(cost_usd=value)


def test_a_written_run_directory_is_readable_by_other_users(tmp_path: Path) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())

    assert run_dir.stat().st_mode & 0o777 == 0o755


def test_a_results_file_that_is_not_utf8_raises_a_schema_error_naming_it(tmp_path: Path) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())
    (run_dir / "results.jsonl").write_bytes(b"\xff\xfe\n")

    with pytest.raises(RecordSchemaError, match=r"results\.jsonl: not valid UTF-8 at byte 0"):
        read_run(run_dir)


def test_a_corrupt_result_line_names_the_failing_field(tmp_path: Path) -> None:
    run_dir = write_run(root=tmp_path, run=make_run())
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").split("\n")
    lines[0] = lines[0].replace('"trial":1', '"trial":0')
    (run_dir / "results.jsonl").write_text("\n".join(lines), encoding="utf-8")

    with pytest.raises(RecordSchemaError, match="trial"):
        read_run(run_dir)
