"""Tests for the static results page."""

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dfo_evals import page as page_module
from dfo_evals.page import (
    CORPUS_CHANGED,
    HEADLINE_CHANGED,
    HISTORY_LIMIT,
    MINUS_SIGN,
    NO_PREVIOUS,
    NO_SHARED_TRIALS,
    RANGE_SEPARATOR,
    UNREADABLE_HEADING,
    build_page,
    main,
)
from dfo_evals.records import (
    CaseResult,
    CheckResult,
    FaultClass,
    Run,
    RunManifest,
    Verdict,
    write_run,
)

STARTED_AT = datetime(2026, 11, 1, 2, 0, tzinfo=UTC)
HEADLINES = {"apis-search": "answer"}
DAC = "corpus/dac.yaml"
IMF = "corpus/imf.yaml"


def make_case(
    *,
    case_id: str,
    passed: bool | None,
    tags: tuple[str, ...],
    trial: int = 1,
    check: str = "answer",
) -> CaseResult:
    """Make a trial; ``passed=None`` makes an infra-excluded trial."""
    verdict = Verdict.PASS if passed else Verdict.FAIL
    return CaseResult(
        case_id=case_id,
        trial=trial,
        input_digest="sha256:input",
        checks=(CheckResult(name=check, verdict=verdict),),
        fault=FaultClass.INFRA if passed is None else None,
        duration_s=1.0,
        cost_usd=None,
        tags=tags,
    )


def save_run(
    root: Path,
    *,
    day: int,
    passes: dict[str, list[bool | None]],
    suite: str = "apis-search",
    dataset_digest: str = "sha256:dataset",
    headline_check: str | None = None,
) -> None:
    """Write a run on day ``day`` with one case per entry in ``passes[tag]``.

    The headline check defaults to the suite's entry in ``HEADLINES``.
    """
    results = tuple(
        make_case(
            case_id=f"{tag}-{index}", passed=passed, tags=(tag,), check=headline_check or "answer"
        )
        for tag, outcomes in passes.items()
        for index, passed in enumerate(outcomes)
    )
    save_results(
        root,
        day=day,
        results=results,
        suite=suite,
        dataset_digest=dataset_digest,
        headline_check=headline_check,
    )


def save_results(
    root: Path,
    *,
    day: int,
    results: tuple[CaseResult, ...],
    suite: str = "apis-search",
    dataset_digest: str = "sha256:dataset",
    headline_check: str | None = None,
) -> None:
    """Write a run of ``results`` on day ``day``."""
    manifest = RunManifest(
        suite=suite,
        run_id=f"night-{day:03d}",
        started_at=STARTED_AT + timedelta(days=day),
        finished_at=None,
        revision=f"rev{day}",
        environment={},
        dataset_digest=dataset_digest,
        headline_check=headline_check or HEADLINES.get(suite),
    )
    write_run(root=root, run=Run(manifest=manifest, results=results))


def group_row(page: str, label: str) -> list[str]:
    """Return the cell texts of the first table row headed ``label``."""
    match = re.search(rf'<tr><th scope="row">{re.escape(label)}</th>(.*?)</tr>', page)
    assert match, f"no row for {label}"
    return re.findall(r"<td[^>]*>(.*?)</td>", match.group(1))


@pytest.fixture
def two_runs(tmp_path: Path) -> Path:
    save_run(tmp_path, day=1, passes={DAC: [True, False, False, False], IMF: [True, True]})
    save_run(tmp_path, day=2, passes={DAC: [True, True, False, False], IMF: [True, False]})
    return tmp_path


def test_the_group_table_lists_all_cases_first_then_tags_sorted(two_runs: Path) -> None:
    page = build_page(root=two_runs).html

    positions = [page.index(f'<th scope="row">{label}</th>') for label in ("All cases", DAC, IMF)]
    assert positions == sorted(positions)


def test_group_rows_show_rates_and_signed_change(two_runs: Path) -> None:
    page = build_page(root=two_runs).html

    assert group_row(page, "All cases") == ["6", "50.0%", "0.0 pp", "0"]
    assert group_row(page, DAC) == ["4", "50.0%", "+25.0 pp", "0"]
    assert group_row(page, IMF) == ["2", "50.0%", f"{MINUS_SIGN}50.0 pp", "0"]
    assert "Cases passing every trial" not in page


def test_a_group_new_in_the_latest_run_is_marked_new(tmp_path: Path) -> None:
    save_run(tmp_path, day=1, passes={DAC: [True]})
    save_run(tmp_path, day=2, passes={DAC: [True], IMF: [False]})

    page = build_page(root=tmp_path).html

    assert group_row(page, IMF)[2] == NO_PREVIOUS


def test_a_group_with_no_scored_trials_shows_n_a(tmp_path: Path) -> None:
    save_run(tmp_path, day=1, passes={DAC: [None]})
    save_run(tmp_path, day=2, passes={DAC: [None]})

    page = build_page(root=tmp_path).html

    first = re.search(r"night-001</code></th>(.*?)</tr>", page)
    second = re.search(r"night-002</code></th>(.*?)</tr>", page)
    assert first
    assert second
    assert re.findall(r"<td[^>]*>(.*?)</td>", first.group(1))[-2:] == ["n/a", NO_PREVIOUS]
    assert re.findall(r"<td[^>]*>(.*?)</td>", second.group(1))[-2:] == ["n/a", "n/a"]


def test_a_change_of_corpus_replaces_the_change_with_a_flag(tmp_path: Path) -> None:
    save_run(tmp_path, day=1, passes={DAC: [True, False]})
    save_run(tmp_path, day=2, passes={DAC: [True]}, dataset_digest="sha256:smaller")

    page = build_page(root=tmp_path).html

    assert group_row(page, "All cases")[2] == CORPUS_CHANGED
    assert group_row(page, DAC)[2] == CORPUS_CHANGED
    history = page[page.index("History of") :]
    newest_row = history[history.index("night-002") :].split("</tr>")[0]
    assert CORPUS_CHANGED in newest_row
    assert 'class="num up"' not in page


def test_history_is_newest_first_and_capped(tmp_path: Path) -> None:
    for day in range(HISTORY_LIMIT + 2):
        save_run(tmp_path, day=day, passes={DAC: [day % 2 == 0]})

    page = build_page(root=tmp_path).html

    assert f"last {HISTORY_LIMIT} runs, newest first" in page
    history = page[page.index("History of") :]
    run_ids = re.findall(r"(night-\d{3})</code></th>", history)
    assert run_ids == [f"night-{day:03d}" for day in reversed(range(2, HISTORY_LIMIT + 2))]
    # The oldest shown run still has a change, computed against a run not shown.
    oldest_row = history[history.index("night-002") :]
    assert "+100.0 pp" in oldest_row.split("</tr>")[0]


def test_hostile_tags_are_escaped(tmp_path: Path) -> None:
    save_run(tmp_path, day=1, passes={"<script>alert(1)</script>": [True]})

    page = build_page(root=tmp_path).html

    assert "<script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_the_page_is_deterministic(two_runs: Path) -> None:
    assert build_page(root=two_runs).html == build_page(root=two_runs).html


def test_the_page_is_a_self_contained_document(two_runs: Path) -> None:
    page = build_page(root=two_runs).html

    assert '<html lang="en-GB">' in page
    assert "<title>DFO eval results</title>" in page
    assert '<meta name="viewport"' in page
    assert "<script" not in page
    assert "http" not in page
    assert "Latest run <code>night-002</code>, started 2026-11-03T02:00+00:00" in page
    assert "History of apis-search, last 2 runs, newest first" in page
    assert "Tags can overlap, so group rows need not add up to All cases." in page


def test_a_single_run_history_says_one_run(tmp_path: Path) -> None:
    save_run(tmp_path, day=1, passes={DAC: [True]})

    page = build_page(root=tmp_path).html

    assert "History of apis-search, 1 run, newest first" in page


def test_the_scoring_note_names_the_headline_check_or_every_check(tmp_path: Path) -> None:
    save_run(tmp_path, day=1, passes={DAC: [True]})
    save_run(tmp_path, day=1, passes={DAC: [True]}, suite="ai-chat")

    page = build_page(root=tmp_path).html

    assert "headline check <code>answer</code> passes" in page
    assert "In the latest run a trial passes when no check fails and at least one passes" in page
    assert page.index("ai-chat") < page.index("apis-search")


def test_a_multi_trial_suite_shows_cases_by_passing_trials(tmp_path: Path) -> None:
    results = tuple(
        make_case(case_id=case_id, passed=passed, tags=(DAC,), trial=trial)
        for case_id, outcomes in {
            "a": (True, True, True),
            "b": (True, False, False),
            "c": (True, True, None),
        }.items()
        for trial, passed in enumerate(outcomes, start=1)
    )
    manifest = RunManifest(
        suite="ai-chat",
        run_id="night-001",
        started_at=STARTED_AT,
        finished_at=None,
        revision="rev1",
        environment={},
        dataset_digest="sha256:dataset",
        headline_check="answer",
    )
    write_run(root=tmp_path, run=Run(manifest=manifest, results=results))

    page = build_page(root=tmp_path).html

    assert f"Cases by passing trials (0{RANGE_SEPARATOR}3)" in page
    assert group_row(page, "All cases") == [
        "8",
        "75.0%",
        NO_PREVIOUS,
        "1 of 2",
        "0: 0 \N{MIDDLE DOT} 1: 1 \N{MIDDLE DOT} 2: 0 \N{MIDDLE DOT} 3: 1",
        "1 in 1 case",
    ]


def test_the_cli_writes_the_page(two_runs: Path, tmp_path: Path) -> None:
    out = tmp_path / "site" / "index.html"

    code = main(["--root", str(two_runs), "--out", str(out)])

    assert code == 0
    assert out.read_text(encoding="utf-8") == build_page(root=two_runs).html


def test_the_cli_errors_on_an_empty_root(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()

    with pytest.raises(ValueError, match="no readable runs"):
        main(["--root", str(root), "--out", str(tmp_path / "index.html")])


def test_a_change_of_headline_check_is_marked_and_each_run_keeps_its_own(tmp_path: Path) -> None:
    save_run(tmp_path, day=1, passes={DAC: [True, False]})
    save_run(tmp_path, day=2, passes={DAC: [True, False]}, headline_check="answer_v2")

    page = build_page(root=tmp_path).html

    assert group_row(page, "All cases")[:3] == ["2", "50.0%", HEADLINE_CHANGED]
    assert "headline check <code>answer_v2</code> passes" in page


def test_an_unreadable_run_is_listed_and_every_other_run_is_shown(tmp_path: Path) -> None:
    save_run(tmp_path, day=1, passes={DAC: [True]})
    save_run(tmp_path, day=1, passes={DAC: [True]}, suite="other")
    manifest = tmp_path / "other" / "night-001" / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"revision"', '"added_field": 1, "revision"'))

    built = build_page(root=tmp_path)
    page = built.html

    assert built.unreadable == ("other/night-001",)
    assert 'id="suite-apis-search"' in page
    assert 'id="suite-other"' not in page
    assert f"<h2>{UNREADABLE_HEADING}</h2>" in page
    assert "<li><code>other/night-001</code></li>" in page


def test_a_root_of_only_unreadable_runs_raises_naming_them(tmp_path: Path) -> None:
    save_run(tmp_path, day=1, passes={DAC: [True]})
    (tmp_path / "apis-search" / "night-001" / "results.jsonl").unlink()

    with pytest.raises(ValueError, match="apis-search/night-001"):
        build_page(root=tmp_path)


def test_the_cli_writes_the_page_and_fails_when_it_left_a_run_out(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    save_run(root, day=1, passes={DAC: [True]})
    save_run(root, day=2, passes={DAC: [True]})
    (root / "apis-search" / "night-002" / "results.jsonl").unlink()
    out = tmp_path / "index.html"

    assert main(["--root", str(root), "--out", str(out)]) == 1
    assert "<li><code>apis-search/night-002</code></li>" in out.read_text(encoding="utf-8")


def history_row(page: str, run_id: str) -> list[str]:
    """Return the cell texts of the history table row for ``run_id``."""
    history = page[page.index("History of") :]
    row = history[history.index(f"{run_id}</code></th>") :].split("</tr>")[0]
    return re.findall(r"<td[^>]*>(.*?)</td>", row)


def test_a_trial_newly_excluded_leaves_the_change_over_the_trials_both_runs_scored(
    tmp_path: Path,
) -> None:
    save_run(tmp_path, day=1, passes={DAC: [True, False, True]})
    save_run(tmp_path, day=2, passes={DAC: [True, None, False]})

    page = build_page(root=tmp_path).html

    # The whole rate reads 66.7% then 50.0%; over DAC-0 and DAC-2 it falls from 100% to 50%.
    expected = f"{MINUS_SIGN}50.0 pp over 2 trials"
    assert group_row(page, "All cases")[1:3] == ["50.0%", expected]
    assert group_row(page, DAC)[2] == expected
    assert history_row(page, "night-002")[-1] == expected
    assert '<td class="num down">' + expected + "</td>" in page


def test_an_excluded_failing_trial_shows_no_false_gain(tmp_path: Path) -> None:
    save_run(tmp_path, day=1, passes={DAC: [True, False]})
    save_run(tmp_path, day=2, passes={DAC: [True, None]})

    page = build_page(root=tmp_path).html

    assert group_row(page, "All cases")[1:3] == ["100.0%", "0.0 pp over 1 trial"]
    assert 'class="num up"' not in page


def test_a_tag_that_gained_a_case_shows_the_change_over_its_shared_trials(tmp_path: Path) -> None:
    save_results(
        tmp_path,
        day=1,
        results=(
            make_case(case_id="a", passed=False, tags=(DAC,)),
            make_case(case_id="b", passed=True, tags=(IMF,)),
        ),
    )
    save_results(
        tmp_path,
        day=2,
        results=(
            make_case(case_id="a", passed=True, tags=(DAC,)),
            make_case(case_id="b", passed=True, tags=(DAC,)),
        ),
    )

    page = build_page(root=tmp_path).html

    assert group_row(page, "All cases")[2] == "+50.0 pp"
    assert group_row(page, DAC)[1:3] == ["100.0%", "+100.0 pp over 1 trial"]
    assert history_row(page, "night-002")[-1] == "+50.0 pp"


def test_a_group_sharing_no_scored_trial_shows_a_flag(tmp_path: Path) -> None:
    save_results(
        tmp_path,
        day=1,
        results=(
            make_case(case_id="a", passed=True, tags=(DAC,)),
            make_case(case_id="b", passed=False, tags=(IMF,)),
        ),
    )
    save_results(
        tmp_path,
        day=2,
        results=(
            make_case(case_id="a", passed=True, tags=(IMF,)),
            make_case(case_id="b", passed=False, tags=(DAC,)),
        ),
    )

    page = build_page(root=tmp_path).html

    assert group_row(page, "All cases")[2] == "0.0 pp"
    assert group_row(page, DAC)[2] == NO_SHARED_TRIALS
    assert group_row(page, IMF)[2] == NO_SHARED_TRIALS


def test_a_history_run_sharing_no_scored_trial_shows_a_flag(tmp_path: Path) -> None:
    save_run(tmp_path, day=1, passes={DAC: [True]})
    save_run(tmp_path, day=2, passes={IMF: [True]})

    page = build_page(root=tmp_path).html

    assert history_row(page, "night-002")[-1] == NO_SHARED_TRIALS


def test_a_change_of_corpus_or_headline_outranks_a_change_of_scored_trials(
    tmp_path: Path,
) -> None:
    save_run(tmp_path, day=1, passes={DAC: [True, False]})
    save_run(tmp_path, day=2, passes={DAC: [True]}, headline_check="answer_v2")
    save_run(tmp_path, day=3, passes={DAC: [True, False]}, dataset_digest="sha256:other")

    page = build_page(root=tmp_path).html

    assert history_row(page, "night-002")[-1] == HEADLINE_CHANGED
    assert history_row(page, "night-003")[-1] == CORPUS_CHANGED


def test_the_footer_says_how_a_change_is_taken(two_runs: Path) -> None:
    page = build_page(root=two_runs).html

    assert (
        "shown only when both runs scored the same corpus by the same headline check. It "
        "compares success rates over the trials both runs scored, and says how many trials "
        "that is when either run scored others."
    ) in page


def test_a_run_that_cannot_be_opened_is_listed_and_every_other_run_is_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_run(tmp_path, day=1, passes={DAC: [True]})
    save_run(tmp_path, day=2, passes={DAC: [True]})
    read_run = page_module.read_run

    def refuse_night_002(directory: Path) -> Run:
        if directory.name == "night-002":
            raise PermissionError(f"permission denied: {directory}")
        return read_run(directory)

    monkeypatch.setattr(page_module, "read_run", refuse_night_002)

    built = build_page(root=tmp_path)

    assert built.unreadable == ("apis-search/night-002",)
    assert "Latest run <code>night-001</code>" in built.html
