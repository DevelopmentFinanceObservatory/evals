import datetime
import json
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from langfuse.api import DatasetItem, DatasetStatus, NotFoundError

from dfo_evals.langfuse_sink import (
    DatasetItemRef,
    DatasetItemSpec,
    InputMismatchError,
    PushIncompleteError,
    PushMismatchError,
    PushSplitError,
    PushTimeoutError,
    UnknownCaseError,
    WrongProjectError,
    check_project,
    connect,
    push_run,
    upsert_items,
    verify_pushed,
)
from dfo_evals.records import CaseResult, CheckResult, FaultClass, Run, RunManifest, Verdict
from dfo_evals.report import input_digest

_PROJECT = "project-1"
_DATASET = "search-core"
_OTHER_DATASET = "search-extra"
_STARTED = datetime.datetime(2026, 11, 1, tzinfo=datetime.UTC)
_PAGE = 2
# Langfuse reads an item run's scores back only within this time of the page's item runs.
_SCORE_WINDOW = datetime.timedelta(days=1)


class _FakeLangfuse:
    """The Langfuse client surface `langfuse_sink` uses, in memory.

    `create_dataset_item` keeps a field's stored value when given `None`, as Langfuse's
    public API does, and stores whole-number floats as ints. `run_experiment` runs the
    task and evaluators on each item as the SDK does and keeps one item run per item, its
    metadata the experiment's and the item's with every value a string, and records each
    `run_experiment` call's datasets in `experiments`. Each item run's experiment id is
    its dataset's run of that name, a UUID, as `dataset_run_items.create` returns it,
    except that the next `failed_links` items fall back to a span id shared only within
    their `run_experiment` call and report no dataset run id, as the SDK does when that
    call fails. The tasks of the next `failed_tasks` items raise: as in the SDK, they
    leave no item run and no item result.
    `api.experiments.list_items` reads one experiment's item runs back newest first, in one
    dataset when given its id, `_PAGE` to a page, with whole-number floats in outputs
    written as ints, as JavaScript writes JSON. Item runs start at `now`.
    `scores` holds every score as Langfuse stores it, one per id and timestamp, so a
    score written twice under one id and timestamp is overwritten and any other second
    write shows as a second score; an evaluator's score and one written without an id
    get a random id. An item run reads back the scores on its observation or on its
    trace alone, one per id, timestamped within `_SCORE_WINDOW` of the page's item runs
    and no later than `now`. `create_score` records each write in `score_writes`.
    The next push loses its first `drop_runs` item runs, and the next `drop_scores`
    score writes are lost, as a failed send would. `hide_scores` keeps scores from
    reading back, and the next `hidden_reads` reads show nothing, as before Langfuse
    shows a push.
    """

    def __init__(self, *, project: str = _PROJECT) -> None:
        self.datasets: dict[str, Any] = {}
        self.items: dict[str, DatasetItem] = {}
        self.item_writes: list[str] = []
        self.runs: list[dict[str, Any]] = []
        self.experiments: list[tuple[str, set[str]]] = []
        self.drop_runs = 0
        self.failed_links = 0
        self.failed_tasks = 0
        self.drop_scores = 0
        self.hide_scores = False
        self.hidden_reads = 0
        self.score_limits: list[int] = []
        self.score_writes: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []
        self.now = _STARTED + datetime.timedelta(hours=1)
        self.api = SimpleNamespace(
            projects=SimpleNamespace(
                get=lambda: SimpleNamespace(data=[SimpleNamespace(id=project)])
            ),
            experiments=SimpleNamespace(list_items=self._list_items),
        )

    def get_dataset(self, name: str) -> Any:
        if name not in self.datasets:
            raise NotFoundError(body=name)
        dataset_id = self.datasets[name]
        return SimpleNamespace(
            id=dataset_id,
            items=[item for item in self.items.values() if item.dataset_id == dataset_id],
        )

    def create_dataset(self, *, name: str) -> None:
        self.datasets[name] = f"dataset-{name}"

    def create_dataset_item(
        self,
        *,
        dataset_name: str,
        id: str,
        input: Any,
        expected_output: Any,
        metadata: Any,
        status: DatasetStatus,
    ) -> DatasetItem:
        self.item_writes.append(id)
        held = self.items.get(id)
        # Langfuse reads a null field as absent and keeps the stored value.
        fields = {
            name: _js_numbers(value)
            if value is not None
            else (getattr(held, name) if held is not None else None)
            for name, value in {
                "input": input,
                "expected_output": expected_output,
                "metadata": metadata,
            }.items()
        }
        self.items[id] = DatasetItem(
            id=id,
            status=status,
            **fields,
            dataset_id=self.datasets[dataset_name],
            dataset_name=dataset_name,
            created_at=_STARTED,
            updated_at=_STARTED,
            media_references=[],
        )
        return self.items[id]

    def run_experiment(
        self,
        *,
        run_name: str,
        data: list[DatasetItem],
        task: Any,
        evaluators: list[Any],
        metadata: dict[str, str],
        **_: Any,
    ) -> Any:
        self.experiments.append((run_name, {item.dataset_name for item in data}))
        fallback_id = _fallback_id(len(self.experiments))
        results = []
        for item in data:
            if self.failed_tasks:
                self.failed_tasks -= 1
                continue
            experiment_id = dataset_run_id = _experiment_id(item.dataset_name, run_name)
            if self.failed_links:
                self.failed_links -= 1
                experiment_id, dataset_run_id = fallback_id, None
            output = task(item=item)
            evaluations = [
                evaluation
                for evaluator in evaluators
                for evaluation in evaluator(
                    input=item.input,
                    output=output,
                    expected_output=item.expected_output,
                    metadata=item.metadata,
                )
            ]
            results.append(SimpleNamespace(output=output, dataset_run_id=dataset_run_id))
            if self.drop_runs:
                self.drop_runs -= 1
                continue
            trace_id, observation_id = f"trace-{len(self.runs)}", f"obs-{len(self.runs)}"
            for evaluation in evaluations:
                self._store_score(
                    id=str(uuid.uuid4()),
                    timestamp=self.now,
                    trace_id=trace_id,
                    observation_id=observation_id,
                    name=evaluation.name,
                    value=evaluation.value,
                    data_type=evaluation.data_type,
                )
            self.runs.append(
                {
                    "experiment": run_name,
                    "experiment_id": experiment_id,
                    "item": item.id,
                    "input": item.input,
                    "dataset_id": item.dataset_id,
                    "trace_id": trace_id,
                    "observation_id": observation_id,
                    "output": json.dumps(output),
                    "metadata": {
                        key: str(value) for key, value in {**metadata, **item.metadata}.items()
                    },
                    "start_time": self.now,
                }
            )
        return SimpleNamespace(item_results=results)

    def create_score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float,
        data_type: str,
        observation_id: str | None = None,
        score_id: str | None = None,
        timestamp: datetime.datetime | None = None,
    ) -> None:
        write = {
            "id": score_id or str(uuid.uuid4()),
            "timestamp": timestamp or self.now,
            "trace_id": trace_id,
            "observation_id": observation_id,
            "name": name,
            "value": value,
            "data_type": data_type,
        }
        self.score_writes.append(write)
        if self.drop_scores:
            self.drop_scores -= 1
            return
        self._store_score(**write)

    def _store_score(self, **score: Any) -> None:
        key = (score["id"], score["timestamp"])
        self.scores = [s for s in self.scores if (s["id"], s["timestamp"]) != key]
        self.scores.append(score)

    def scores_of(self, run: dict[str, Any]) -> list[dict[str, Any]]:
        """Every score Langfuse holds on an item run, duplicates included."""
        return [
            s
            for s in self.scores
            if s["observation_id"] == run["observation_id"]
            or (s["trace_id"] == run["trace_id"] and s["observation_id"] is None)
        ]

    def flush(self) -> None:
        pass

    def _list_items(
        self,
        *,
        from_start_time: datetime.datetime,
        experiment_name: str,
        score_limit: int,
        cursor: str | None,
        dataset_id: str | None = None,
        **_: Any,
    ) -> Any:
        assert from_start_time == _STARTED
        self.score_limits.append(score_limit)
        if self.hidden_reads:
            self.hidden_reads -= 1
            return SimpleNamespace(data=[], meta=SimpleNamespace(cursor=None))
        shown = [
            run
            for run in reversed(self.runs)
            if run["experiment"] == experiment_name and dataset_id in (None, run["dataset_id"])
        ]
        start = int(cursor or 0)
        page = shown[start : start + _PAGE]
        if page:
            earliest = min(run["start_time"] for run in page) - _SCORE_WINDOW
            latest = min(self.now, max(run["start_time"] for run in page) + _SCORE_WINDOW)
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    id=run["observation_id"],
                    trace_id=run["trace_id"],
                    start_time=run["start_time"],
                    experiment_id=run["experiment_id"],
                    experiment_item_id=run["item"],
                    input=json.dumps(_js_numbers(run["input"])),
                    output=json.dumps(_js_numbers(json.loads(run["output"]))),
                    metadata=run["metadata"],
                    scores=[]
                    if self.hide_scores
                    else [
                        SimpleNamespace(name=score["name"], value=score["value"])
                        for score in {s["id"]: s for s in self.scores_of(run)}.values()
                        if earliest <= score["timestamp"] <= latest
                    ],
                )
                for run in page
            ],
            meta=SimpleNamespace(cursor=str(start + _PAGE) if start + _PAGE < len(shown) else None),
        )

    def runs_of(self, case_id: str) -> list[dict[str, Any]]:
        return [run for run in self.runs if run["metadata"]["case_id"] == case_id]


def _score_values(fake: _FakeLangfuse, run: dict[str, Any]) -> dict[str, float]:
    """The value of each score Langfuse holds on an item run, by name."""
    return {score["name"]: score["value"] for score in fake.scores_of(run)}


def _experiment_id(dataset: str, run_name: str) -> str:
    """The id of a dataset's run of a name, a UUID, as linking an item run to it returns."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{dataset}/{run_name}"))


def _fallback_id(call: int) -> str:
    """The id the SDK falls back to in its `call`th `run_experiment` call: a span id."""
    return f"{call:016x}"


def _js_numbers(value: Any) -> Any:
    """`value` as JavaScript writes it back to JSON: whole-number floats as ints."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _js_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_js_numbers(item) for item in value]
    return value


def _result(
    case_id: str,
    trial: int = 1,
    *,
    duration_s: float = 1.5,
    cost_usd: float = 0.01,
    **verdicts: Verdict,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        trial=trial,
        input_digest=input_digest(_input(case_id)),
        checks=tuple(CheckResult(name=name, verdict=v) for name, v in verdicts.items()),
        fault=FaultClass.AGENT if Verdict.FAIL in verdicts.values() else None,
        duration_s=duration_s,
        cost_usd=cost_usd,
        trace_id=f"sut-{case_id}-{trial}",
        tags=("core",),
    )


def _run(*results: CaseResult, run_id: str = "20261101T000000Z") -> Run:
    return Run(
        manifest=RunManifest(
            suite="search",
            run_id=run_id,
            started_at=_STARTED,
            finished_at=None,
            revision="abc123",
            environment={},
            dataset_digest="sha256:dataset",
        ),
        results=results,
    )


def _input(case_id: str) -> dict[str, str]:
    """The input the run evaluated case `case_id` on."""
    return {"q": case_id}


def _item_id(case_id: str, dataset: str = _DATASET) -> str:
    return f"{dataset}:search-case-{case_id}"


def _item(case_id: str) -> DatasetItemRef:
    return DatasetItemRef(dataset=_DATASET, item_id=_item_id(case_id))


def _split_item(case_id: str) -> DatasetItemRef:
    """Case "a" in `_DATASET`, every other case in `_OTHER_DATASET`."""
    dataset = _DATASET if case_id == "a" else _OTHER_DATASET
    return DatasetItemRef(dataset=dataset, item_id=_item_id(case_id, dataset))


def _add_items(fake: _FakeLangfuse, *case_ids: str, dataset: str = _DATASET) -> None:
    upsert_items(
        client=fake,
        dataset=dataset,
        items=[DatasetItemSpec(item_id=_item_id(c, dataset), input=_input(c)) for c in case_ids],
    )
    fake.item_writes.clear()


def _with_items(*case_ids: str) -> _FakeLangfuse:
    fake = _FakeLangfuse()
    _add_items(fake, *case_ids)
    return fake


def _push(fake: _FakeLangfuse, run: Run, *, timeout_s: float = 0, **kwargs: Any) -> str:
    kwargs.setdefault("item_of", _item)
    return push_run(client=fake, run=run, timeout_s=timeout_s, **kwargs)


def _verify(fake: _FakeLangfuse, run: Run, name: str, **kwargs: Any) -> None:
    kwargs.setdefault("item_of", _item)
    verify_pushed(client=fake, run=run, run_name=name, **kwargs)


_TWO_CASES = _run(
    _result("a", answer=Verdict.PASS, rows=Verdict.FAIL, units=Verdict.UNVERIFIABLE),
    _result("b", answer=Verdict.NOT_APPLICABLE, rows=Verdict.UNREACHED),
)


def test_a_client_of_another_project_is_refused() -> None:
    check_project(client=_FakeLangfuse(), project_id=_PROJECT)

    with pytest.raises(WrongProjectError, match="another"):
        check_project(
            client=_FakeLangfuse(project="another"),
            project_id=_PROJECT,
        )


def test_connect_refuses_an_empty_credential() -> None:
    with pytest.raises(ValueError, match="secret_key"):
        connect(public_key="pk", secret_key="", base_url="http://127.0.0.1:9", project_id=_PROJECT)


def test_connect_refuses_keys_of_another_project(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[dict[str, Any]] = []

    def client(**kwargs: Any) -> _FakeLangfuse:
        built.append(kwargs)
        return _FakeLangfuse(project="another")

    monkeypatch.setattr("dfo_evals.langfuse_sink.Langfuse", client)

    with pytest.raises(WrongProjectError):
        connect(public_key="pk", secret_key="sk", base_url="http://lf", project_id=_PROJECT)
    assert built[0]["tracer_provider"] is not None


def test_items_are_created_with_their_dataset_and_unchanged_items_are_not_rewritten() -> None:
    fake = _FakeLangfuse()
    items = [
        DatasetItemSpec(
            item_id="x:1", input={"q": "one"}, expected_output=["a"], metadata={"k": 1}
        ),
        DatasetItemSpec(item_id="x:2", input={"q": "two"}),
    ]

    created = upsert_items(client=fake, dataset="x", items=items)
    unchanged = upsert_items(client=fake, dataset="x", items=items)
    changed = upsert_items(
        client=fake,
        dataset="x",
        items=[items[0], DatasetItemSpec(item_id="x:2", input={"q": "changed"})],
    )

    assert set(fake.datasets) == {"x"}
    assert (created, unchanged, changed) == (["x:1", "x:2"], [], ["x:2"])
    assert fake.item_writes == ["x:1", "x:2", "x:2"]
    assert fake.items["x:1"].expected_output == ["a"]


def test_a_none_field_keeps_the_stored_value_and_a_second_sync_writes_nothing() -> None:
    fake = _FakeLangfuse()
    full = DatasetItemSpec(
        item_id="x:1", input={"q": 1.0}, expected_output=["a"], metadata={"k": 2.0}
    )
    upsert_items(client=fake, dataset="x", items=[full])
    bare = DatasetItemSpec(item_id="x:1", input={"q": 1})

    first = upsert_items(client=fake, dataset="x", items=[bare])
    second = upsert_items(client=fake, dataset="x", items=[bare])
    again = upsert_items(client=fake, dataset="x", items=[full])

    assert (first, second, again) == ([], [], [])
    assert (fake.items["x:1"].expected_output, fake.items["x:1"].metadata) == (["a"], {"k": 2})


def test_a_field_replaced_by_another_value_is_written_once() -> None:
    fake = _FakeLangfuse()
    upsert_items(
        client=fake,
        dataset="x",
        items=[DatasetItemSpec(item_id="x:1", input={"q": 1}, metadata={"k": 1})],
    )
    emptied = [DatasetItemSpec(item_id="x:1", input={"q": 1}, metadata={})]

    assert upsert_items(client=fake, dataset="x", items=emptied) == ["x:1"]
    assert upsert_items(client=fake, dataset="x", items=emptied) == []
    assert fake.items["x:1"].metadata == {}


def test_an_archived_item_is_written_again_as_active() -> None:
    fake = _FakeLangfuse()
    spec = DatasetItemSpec(item_id="x:1", input={"q": "one"})
    upsert_items(client=fake, dataset="x", items=[spec])
    fake.items["x:1"] = fake.items["x:1"].model_copy(update={"status": DatasetStatus.ARCHIVED})

    assert upsert_items(client=fake, dataset="x", items=[spec]) == ["x:1"]
    assert fake.items["x:1"].status == DatasetStatus.ACTIVE


def test_pass_scores_one_fail_scores_zero_and_other_verdicts_are_not_scored() -> None:
    fake = _with_items("a", "b")

    name = _push(fake, _TWO_CASES)

    assert name.startswith("search/20261101T000000Z abc123 ")
    (a,) = fake.runs_of("a")
    (b,) = fake.runs_of("b")
    assert a["item"] == _item_id("a")
    assert _score_values(fake, a) == {"answer": 1.0, "rows": 0.0}
    assert {score["data_type"] for score in fake.scores_of(a)} == {"NUMERIC"}
    assert _score_values(fake, b) == {}


def test_every_trial_of_a_case_goes_in_one_item_run_scored_as_its_pass_share() -> None:
    run = _run(
        _result("a", 1, answer=Verdict.PASS, rows=Verdict.PASS),
        _result("a", 2, answer=Verdict.FAIL, rows=Verdict.UNVERIFIABLE),
        _result("a", 3, answer=Verdict.PASS, rows=Verdict.UNREACHED),
        _result("a", 4, answer=Verdict.PASS, rows=Verdict.PASS),
    )
    fake = _with_items("a")

    _push(fake, run, outputs={("a", 1): {"answer": 42}, ("a", 3): "text"})

    (a,) = fake.runs_of("a")
    assert _score_values(fake, a) == {"answer": 0.75, "rows": 1.0}
    output = json.loads(a["output"])
    assert [r["trial"] for r in output["results"]] == [1, 2, 3, 4]
    assert output["outputs"] == [{"answer": 42}, None, "text", None]
    trials = json.loads(a["metadata"]["trials"])
    assert trials[1] == {
        "trial": 2,
        "fault": "agent",
        "duration_s": 1.5,
        "cost_usd": 0.01,
        "tags": ["core"],
        "input_digest": input_digest(_input("a")),
        "trace_id": "sut-a-2",
    }


def test_a_case_with_no_dataset_item_raises_before_anything_is_written() -> None:
    fake = _with_items("a")

    with pytest.raises(UnknownCaseError, match="'b'"):
        _push(fake, _TWO_CASES)

    assert fake.runs == []
    assert fake.item_writes == []


def test_a_push_to_a_missing_dataset_raises_and_creates_nothing() -> None:
    fake = _FakeLangfuse()

    with pytest.raises(UnknownCaseError):
        _push(fake, _TWO_CASES)

    assert fake.datasets == {}


def test_a_run_over_two_datasets_pushes_each_case_to_its_own_dataset_under_one_name() -> None:
    fake = _with_items("a")
    _add_items(fake, "b", dataset=_OTHER_DATASET)

    name = _push(fake, _TWO_CASES, item_of=_split_item)

    (a,) = fake.runs_of("a")
    (b,) = fake.runs_of("b")
    assert (a["item"], a["dataset_id"]) == (_item_id("a"), fake.datasets[_DATASET])
    assert (b["item"], b["dataset_id"]) == (
        _item_id("b", _OTHER_DATASET),
        fake.datasets[_OTHER_DATASET],
    )
    assert fake.experiments == [(name, {_DATASET}), (name, {_OTHER_DATASET})]
    _verify(fake, _TWO_CASES, name, item_of=_split_item, timeout_s=0)

    assert _push(fake, _TWO_CASES, item_of=_split_item) == name
    assert (len(fake.runs), len(fake.experiments)) == (2, 2)


def test_a_case_lost_in_one_of_two_datasets_is_sent_again_to_that_dataset_only() -> None:
    fake = _with_items("a")
    _add_items(fake, "b", dataset=_OTHER_DATASET)
    # The first item run sent, case "a" in _DATASET, is lost.
    fake.drop_runs = 1

    with pytest.raises(PushIncompleteError, match=r"\['a'\]"):
        _push(fake, _TWO_CASES, item_of=_split_item)
    name = _push(fake, _TWO_CASES, item_of=_split_item)

    assert fake.experiments[2:] == [(name, {_DATASET})]
    assert sorted(run["metadata"]["case_id"] for run in fake.runs) == ["a", "b"]
    _verify(fake, _TWO_CASES, name, item_of=_split_item, timeout_s=0)


def test_a_case_mapped_to_an_item_its_dataset_lacks_raises_naming_the_dataset() -> None:
    fake = _with_items("a", "b")
    fake.create_dataset(name=_OTHER_DATASET)

    with pytest.raises(UnknownCaseError, match=rf"dataset '{_OTHER_DATASET}'.*\['b'\]"):
        _push(fake, _TWO_CASES, item_of=_split_item)

    assert (fake.runs, fake.item_writes) == ([], [])


def test_a_case_shown_in_a_dataset_it_does_not_map_to_is_a_mismatch() -> None:
    fake = _with_items("a", "b")
    _add_items(fake, "b", dataset=_OTHER_DATASET)
    name = _push(fake, _TWO_CASES)

    with pytest.raises(PushMismatchError, match=rf"\['b'\].*'{_DATASET}'"):
        _verify(fake, _TWO_CASES, name, item_of=_split_item, timeout_s=0)


def test_the_same_run_pushed_to_another_dataset_is_written_and_verified_there() -> None:
    fake = _with_items("a", "b")
    _add_items(fake, "a", "b", dataset=_OTHER_DATASET)

    def other(case_id: str) -> DatasetItemRef:
        return DatasetItemRef(dataset=_OTHER_DATASET, item_id=_item_id(case_id, _OTHER_DATASET))

    name = _push(fake, _TWO_CASES)
    assert _push(fake, _TWO_CASES, item_of=other) == name

    other_id = fake.datasets[_OTHER_DATASET]
    assert sorted(run["item"] for run in fake.runs if run["dataset_id"] == other_id) == [
        _item_id("a", _OTHER_DATASET),
        _item_id("b", _OTHER_DATASET),
    ]
    assert fake.experiments == [(name, {_DATASET}), (name, {_OTHER_DATASET})]
    _verify(fake, _TWO_CASES, name, item_of=other, timeout_s=0)


def test_an_item_input_changed_after_the_run_is_refused_before_writing() -> None:
    fake = _with_items("a")
    _add_items(fake, "b", dataset=_OTHER_DATASET)
    upsert_items(
        client=fake,
        dataset=_OTHER_DATASET,
        items=[DatasetItemSpec(item_id=_item_id("b", _OTHER_DATASET), input={"q": "edited"})],
    )
    fake.item_writes.clear()

    with pytest.raises(InputMismatchError, match=rf"dataset '{_OTHER_DATASET}' cases \['b'\]"):
        _push(fake, _TWO_CASES, item_of=_split_item)

    assert (fake.runs, fake.experiments, fake.item_writes) == ([], [], [])


def test_an_item_input_changed_after_a_push_fails_verification() -> None:
    fake = _with_items("a", "b")
    name = _push(fake, _TWO_CASES)
    upsert_items(
        client=fake,
        dataset=_DATASET,
        items=[DatasetItemSpec(item_id=_item_id("a"), input={"q": "edited"})],
    )

    with pytest.raises(InputMismatchError, match=r"\['a'\]"):
        _verify(fake, _TWO_CASES, name, timeout_s=0)


def test_a_run_that_digested_a_whole_float_matches_an_item_read_back_as_an_int() -> None:
    fake = _FakeLangfuse()
    upsert_items(
        client=fake,
        dataset=_DATASET,
        items=[DatasetItemSpec(item_id=_item_id("a"), input={"q": "a", "n": 1})],
    )
    run = _run(
        _result("a", answer=Verdict.PASS).model_copy(
            update={"input_digest": input_digest({"q": "a", "n": 1.0})}
        )
    )

    name = _push(fake, run)

    _verify(fake, run, name, timeout_s=0)


def test_an_item_run_on_another_input_fails_verification() -> None:
    fake = _with_items("a", "b")
    name = _push(fake, _TWO_CASES)
    fake.runs_of("a")[0]["input"] = {"q": "other"}

    with pytest.raises(PushMismatchError, match="another input"):
        _verify(fake, _TWO_CASES, name, timeout_s=0)


def test_an_item_run_of_another_item_fails_verification() -> None:
    fake = _with_items("a", "b")
    name = _push(fake, _TWO_CASES)
    fake.runs_of("a")[0]["item"] = _item_id("b")

    with pytest.raises(PushMismatchError, match=r"item run of .*search-case-b"):
        _verify(fake, _TWO_CASES, name, timeout_s=0)


def test_trials_of_one_case_with_different_input_digests_are_refused() -> None:
    fake = _with_items("a")
    run = _run(
        _result("a", 1, answer=Verdict.PASS),
        _result("a", 2, answer=Verdict.PASS).model_copy(update={"input_digest": "sha256:x"}),
    )

    with pytest.raises(ValueError, match="different input digests"):
        _push(fake, run)

    assert fake.runs == []


def test_two_cases_mapped_to_one_item_are_refused() -> None:
    fake = _with_items("a", "b")

    with pytest.raises(ValueError, match="same dataset item"):
        _push(fake, _TWO_CASES, item_of=lambda _: _item("a"))

    assert fake.runs == []


def test_a_push_with_an_item_run_outside_its_dataset_run_raises_and_fails_verify() -> None:
    fake = _with_items("a", "b")
    fake.failed_links = 1

    with pytest.raises(PushSplitError, match=rf"dataset '{_DATASET}' were not linked"):
        _push(fake, _TWO_CASES)
    (name,) = {run["experiment"] for run in fake.runs}

    with pytest.raises(PushSplitError, match=_fallback_id(1)):
        _verify(fake, _TWO_CASES, name, timeout_s=0)
    with pytest.raises(PushSplitError):
        _push(fake, _TWO_CASES)
    assert len(fake.runs) == 2


def test_a_case_sent_again_lands_in_the_experiment_of_the_first_push() -> None:
    fake = _with_items("a", "b")
    fake.drop_runs = 1

    with pytest.raises(PushIncompleteError):
        _push(fake, _TWO_CASES)
    name = _push(fake, _TWO_CASES)

    assert len(fake.experiments) == 2
    assert {run["experiment_id"] for run in fake.runs} == {_experiment_id(_DATASET, name)}
    _verify(fake, _TWO_CASES, name, timeout_s=0)


def test_a_case_sent_again_outside_the_first_experiment_raises() -> None:
    fake = _with_items("a", "b")
    fake.drop_runs = 1
    with pytest.raises(PushIncompleteError):
        _push(fake, _TWO_CASES)
    fake.failed_links = 1

    with pytest.raises(PushSplitError, match="not linked"):
        _push(fake, _TWO_CASES)
    with pytest.raises(PushSplitError, match=_fallback_id(2)):
        _verify(fake, _TWO_CASES, fake.runs[0]["experiment"], timeout_s=0)


def test_a_push_whose_every_item_run_falls_back_raises_and_so_do_retries() -> None:
    fake = _with_items("a", "b")
    fake.failed_links = 2

    with pytest.raises(PushSplitError, match="not linked to the dataset run"):
        _push(fake, _TWO_CASES)
    assert {run["experiment_id"] for run in fake.runs} == {_fallback_id(1)}
    (name,) = {run["experiment"] for run in fake.runs}

    with pytest.raises(PushSplitError, match=rf"not linked .*{_fallback_id(1)}"):
        _push(fake, _TWO_CASES)
    with pytest.raises(PushSplitError, match=rf"not linked .*{_fallback_id(1)}"):
        _verify(fake, _TWO_CASES, name, timeout_s=0)
    assert (len(fake.runs), len(fake.experiments), fake.score_writes) == (2, 1, [])


def test_a_split_in_the_first_of_two_datasets_raises_and_so_does_a_retry() -> None:
    run = _run(
        _result("a", answer=Verdict.PASS),
        _result("c", answer=Verdict.FAIL),
        _result("b", answer=Verdict.PASS),
    )
    fake = _with_items("a", "c")
    _add_items(fake, "b", dataset=_OTHER_DATASET)

    def item_of(case_id: str) -> DatasetItemRef:
        dataset = _OTHER_DATASET if case_id == "b" else _DATASET
        return DatasetItemRef(dataset=dataset, item_id=_item_id(case_id, dataset))

    # Case "a", the first item sent to the first dataset, falls back.
    fake.failed_links = 1
    fake.drop_scores = 2
    with pytest.raises(PushSplitError, match=rf"dataset '{_DATASET}'"):
        _push(fake, run, item_of=item_of)

    with pytest.raises(PushSplitError, match=rf"dataset '{_DATASET}'"):
        _push(fake, run, item_of=item_of)
    with pytest.raises(PushSplitError, match=rf"dataset '{_DATASET}'"):
        _verify(fake, run, fake.runs[0]["experiment"], item_of=item_of, timeout_s=0)
    assert fake.experiments[1:] == []
    assert fake.score_writes == []


def test_an_item_whose_task_raised_is_left_out_and_reported_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = _with_items("a", "b")
    fake.failed_tasks = 1

    with pytest.raises(PushIncompleteError, match=r"shows 1 of the 2 cases.*\['a'\]"):
        _push(fake, _TWO_CASES)

    assert "run_experiment returned 1 of 2 item runs" in caplog.text


def test_outputs_for_a_trial_the_run_lacks_are_refused() -> None:
    with pytest.raises(ValueError, match="lacks"):
        _push(_with_items("a", "b"), _TWO_CASES, outputs={("a", 2): "x"})


def test_a_push_whose_scores_were_lost_raises_and_pushing_again_fills_them() -> None:
    fake = _with_items("a", "b")
    fake.drop_scores = 2

    with pytest.raises(PushTimeoutError, match="pushing the run again writes them again"):
        _push(fake, _TWO_CASES)
    assert fake.scores == []
    name = _push(fake, _TWO_CASES)

    assert len(fake.runs) == 2
    assert _score_values(fake, fake.runs_of("a")[0]) == {"answer": 1.0, "rows": 0.0}
    _verify(fake, _TWO_CASES, name, timeout_s=0)


def test_a_score_repaired_days_after_the_push_reads_back_and_verifies() -> None:
    fake = _with_items("a", "b")
    fake.drop_scores = 2
    with pytest.raises(PushTimeoutError):
        _push(fake, _TWO_CASES)
    fake.now += datetime.timedelta(days=2)

    name = _push(fake, _TWO_CASES)

    assert {write["timestamp"] for write in fake.score_writes} == {
        run["start_time"] for run in fake.runs_of("a")
    }
    _verify(fake, _TWO_CASES, name, timeout_s=0)


def test_a_score_repaired_twice_is_written_under_the_same_id_and_time() -> None:
    fake = _with_items("a", "b")
    fake.drop_scores = 2
    with pytest.raises(PushTimeoutError):
        _push(fake, _TWO_CASES)
    fake.hide_scores = True

    for _ in range(2):
        with pytest.raises(PushTimeoutError):
            _push(fake, _TWO_CASES)

    first, second, third = (fake.score_writes[i : i + 2] for i in (0, 2, 4))
    assert len(fake.score_writes) == 6
    assert [w["name"] for w in first] == ["answer", "rows"]
    assert first == second == third
    assert len({w["id"] for w in first}) == 2


def test_a_late_score_written_again_stays_one_score() -> None:
    fake = _with_items("a", "b")
    # The scores land but do not show before the push gives up, nor on the next push.
    fake.hide_scores = True
    with pytest.raises(PushTimeoutError):
        _push(fake, _TWO_CASES)
    with pytest.raises(PushTimeoutError):
        _push(fake, _TWO_CASES)
    fake.hide_scores = False

    (a,) = fake.runs_of("a")
    name = a["experiment"]
    assert len(fake.score_writes) == 4
    assert sorted(score["name"] for score in fake.scores_of(a)) == ["answer", "rows"]
    _verify(fake, _TWO_CASES, name, timeout_s=0)


def test_every_score_is_written_once_on_the_item_run_and_a_repeat_push_writes_none() -> None:
    fake = _with_items("a", "b")

    name = _push(fake, _TWO_CASES)
    assert _push(fake, _TWO_CASES) == name

    (a,) = fake.runs_of("a")
    assert [(w["name"], w["trace_id"], w["observation_id"]) for w in fake.score_writes] == [
        ("answer", a["trace_id"], a["observation_id"]),
        ("rows", a["trace_id"], a["observation_id"]),
    ]
    assert {w["timestamp"] for w in fake.score_writes} == {a["start_time"]}


def test_a_case_that_never_appears_raises_and_pushing_again_sends_only_it() -> None:
    fake = _with_items("a", "b")
    fake.drop_runs = 1

    with pytest.raises(PushIncompleteError, match=r"shows 1 of the 2 cases.*push the run again"):
        _push(fake, _TWO_CASES)
    name = _push(fake, _TWO_CASES)

    assert sorted(run["metadata"]["case_id"] for run in fake.runs) == ["a", "b"]
    _verify(fake, _TWO_CASES, name, timeout_s=0)


def test_a_push_waits_for_langfuse_to_show_it(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _with_items("a", "b")
    # The read before sending and the first read after it show nothing.
    fake.hidden_reads = 2
    sleeps: list[float] = []
    monkeypatch.setattr("dfo_evals.langfuse_sink.time.sleep", sleeps.append)

    name = _push(fake, _TWO_CASES, timeout_s=60)

    assert len(sleeps) == 1
    assert len(fake.runs) == 2
    assert _push(fake, _TWO_CASES) == name
    assert len(fake.runs) == 2


def test_whole_number_floats_read_back_as_ints_still_match() -> None:
    run = _run(
        _result("a", duration_s=2.0, cost_usd=0.0, answer=Verdict.PASS),
        _result("b", duration_s=0.5, cost_usd=0.0, answer=Verdict.FAIL),
    )
    fake = _with_items("a", "b")
    outputs = {("a", 1): {"total": 3.0, "share": 0.5}}

    name = _push(fake, run, outputs=outputs)

    assert (
        '"cost_usd": 0,'
        in fake._list_items(
            from_start_time=_STARTED, experiment_name=name, score_limit=50, cursor=None
        )
        .data[-1]
        .output
    )
    _verify(fake, run, name, timeout_s=0)
    assert _push(fake, run, outputs=outputs) == name
    assert len(fake.runs) == 2


def test_scores_are_read_back_up_to_the_api_cap_and_a_case_over_it_is_refused() -> None:
    fake = _with_items("a", "b")
    _push(fake, _TWO_CASES)
    assert set(fake.score_limits) == {50}

    over = _run(_result("a", **{f"check_{i}": Verdict.PASS for i in range(51)}))
    with pytest.raises(ValueError, match="more than 50"):
        _push(_with_items("a"), over)


def test_an_item_with_no_input_is_refused_before_anything_is_written() -> None:
    fake = _FakeLangfuse()

    with pytest.raises(ValueError, match="no input"):
        upsert_items(
            client=fake,
            dataset="x",
            items=[
                DatasetItemSpec(item_id="x:1", input={"q": 1}),
                DatasetItemSpec(item_id="x:2", input=None),
            ],
        )

    assert (fake.datasets, fake.item_writes) == ({}, [])


def test_a_push_over_a_dataset_item_with_no_input_is_refused_before_writing() -> None:
    fake = _with_items("a")
    fake.create_dataset_item(
        dataset_name=_DATASET,
        id=_item_id("b"),
        input=None,
        expected_output=None,
        metadata=None,
        status=DatasetStatus.ACTIVE,
    )

    with pytest.raises(ValueError, match="no input"):
        _push(fake, _TWO_CASES)

    assert fake.runs == []


def test_a_different_run_goes_up_as_a_different_experiment() -> None:
    fake = _with_items("a", "b")
    other = _run(*_TWO_CASES.results[:1])

    names = {
        _push(fake, _TWO_CASES),
        _push(fake, other),
        _push(fake, _TWO_CASES, outputs={("a", 1): "x"}),
    }

    assert len(names) == 3


def test_a_case_edited_in_langfuse_is_refused_on_push_again() -> None:
    fake = _with_items("a", "b")
    _push(fake, _TWO_CASES)
    fake.runs_of("a")[0]["metadata"]["case_digest"] = "sha256:other"

    with pytest.raises(PushMismatchError, match="'a'"):
        _push(fake, _TWO_CASES)


def test_verify_pushed_passes_on_a_complete_push() -> None:
    fake = _with_items("a", "b")
    name = _push(fake, _TWO_CASES)

    _verify(fake, _TWO_CASES, name, timeout_s=0)


def test_verify_pushed_raises_when_a_score_is_missing() -> None:
    fake = _with_items("a", "b")
    name = _push(fake, _TWO_CASES)
    fake.scores = [s for s in fake.scores if s["name"] != "rows"]

    with pytest.raises(PushTimeoutError, match=r"case 'a' scores \['rows'\]"):
        _verify(fake, _TWO_CASES, name, timeout_s=0)


def test_verify_pushed_raises_when_a_score_differs() -> None:
    fake = _with_items("a", "b")
    name = _push(fake, _TWO_CASES)
    next(s for s in fake.scores if s["name"] == "rows")["value"] = 1.0

    with pytest.raises(PushMismatchError, match=r"score 'rows' is 1\.0"):
        _verify(fake, _TWO_CASES, name, timeout_s=0)


def test_verify_pushed_raises_when_an_output_was_edited() -> None:
    fake = _with_items("a", "b")
    name = _push(fake, _TWO_CASES)
    run = fake.runs_of("b")[0]
    run["output"] = run["output"].replace('"b"', '"c"', 1)

    with pytest.raises(PushMismatchError, match="output"):
        _verify(fake, _TWO_CASES, name, timeout_s=0)


def test_verify_pushed_raises_on_timeout_when_langfuse_never_shows_the_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _with_items("a", "b")
    name = _push(fake, _TWO_CASES)
    fake.hide_scores = True
    sleeps: list[float] = []
    clock = iter([0.0, 0.0, 0.05, 0.2])
    monkeypatch.setattr("dfo_evals.langfuse_sink.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("dfo_evals.langfuse_sink.time.sleep", sleeps.append)

    with pytest.raises(PushTimeoutError, match="after 0 s"):
        _verify(fake, _TWO_CASES, name, timeout_s=0.1)

    assert sleeps == [pytest.approx(0.1), pytest.approx(0.05)]
