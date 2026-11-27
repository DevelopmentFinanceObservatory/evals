"""Tests for scoring run records."""

from datetime import UTC, datetime, timedelta

import pytest

from dfo_evals.records import (
    TASK_CHECK_NAME,
    CaseResult,
    CheckResult,
    FaultClass,
    Run,
    RunManifest,
    Verdict,
)
from dfo_evals.summary import (
    GroupSummary,
    MissingHeadlineCheckError,
    SharedChange,
    TrialOutcome,
    chronological,
    shared_change,
    summarise_run,
    trial_outcome,
)

STARTED_AT = datetime(2026, 11, 24, 2, 0, tzinfo=UTC)
HEADLINE = "answer"


def make_case(
    *,
    case_id: str = "case-1",
    trial: int = 1,
    verdicts: dict[str, Verdict] | None = None,
    fault: FaultClass | None = None,
    tags: tuple[str, ...] = (),
) -> CaseResult:
    verdicts = {HEADLINE: Verdict.PASS} if verdicts is None else verdicts
    return CaseResult(
        case_id=case_id,
        trial=trial,
        input_digest="sha256:input",
        checks=tuple(CheckResult(name=name, verdict=v) for name, v in verdicts.items()),
        fault=fault,
        duration_s=1.0,
        cost_usd=None,
        tags=tags,
    )


def make_run(
    *results: CaseResult,
    run_id: str = "run-1",
    started_at: datetime = STARTED_AT,
    headline_check: str | None = HEADLINE,
) -> Run:
    manifest = RunManifest(
        suite="apis-search",
        run_id=run_id,
        started_at=started_at,
        finished_at=None,
        revision="abc123",
        environment={},
        dataset_digest="sha256:dataset",
        headline_check=headline_check,
    )
    return Run(manifest=manifest, results=results)


def group(**overrides) -> GroupSummary:
    """A consistent single-trial group of two cases, one passing; override to break it."""
    fields = {
        "trials_per_case": 1,
        "cases": 2,
        "scored_trials": 2,
        "passes": 1,
        "excluded_trials": 0,
        "cases_with_excluded_trial": 0,
        "fully_scored_passes": 1,
        "cases_passing_all": 1,
    } | overrides
    outcomes = scored_outcomes(fields["scored_trials"], passes=fields["passes"])
    return GroupSummary(**(fields | {"scored_outcomes": outcomes}))


def scored_outcomes(
    count: int, *, passes: int, trials_per_case: int = 1
) -> frozenset[tuple[str, int, bool]]:
    """``count`` distinct ``(case_id, trial, passed)`` outcomes, the first ``passes`` passing."""
    return frozenset(
        (f"case-{index // trials_per_case}", index % trials_per_case + 1, index < passes)
        for index in range(count)
    )


def multi_group(**overrides) -> GroupSummary:
    """A consistent two-trial group: one case passing twice, one passing once."""
    fields = {
        "trials_per_case": 2,
        "cases": 2,
        "scored_trials": 4,
        "passes": 3,
        "excluded_trials": 0,
        "cases_with_excluded_trial": 0,
        "fully_scored_passes": 3,
        "cases_passing_all": 1,
        "pass_distribution": (0, 1, 1),
    } | overrides
    outcomes = scored_outcomes(
        fields["scored_trials"], passes=fields["passes"], trials_per_case=fields["trials_per_case"]
    )
    return GroupSummary(**(fields | {"scored_outcomes": outcomes}))


@pytest.mark.parametrize("fault", [FaultClass.GRADER, FaultClass.INFRA, FaultClass.REFERENCE])
def test_grader_infra_and_reference_faults_exclude_the_trial(fault: FaultClass) -> None:
    result = make_case(verdicts={}, fault=fault)

    assert trial_outcome(result=result, headline_check=HEADLINE) is TrialOutcome.EXCLUDED


@pytest.mark.parametrize("fault", [FaultClass.AGENT, FaultClass.UPSTREAM])
def test_agent_and_upstream_faults_count_as_fails(fault: FaultClass) -> None:
    result = make_case(verdicts={HEADLINE: Verdict.FAIL}, fault=fault)

    assert trial_outcome(result=result, headline_check=HEADLINE) is TrialOutcome.FAIL


@pytest.mark.parametrize("fault", [FaultClass.AGENT, FaultClass.UPSTREAM])
def test_agent_and_upstream_faults_fail_a_trial_with_a_passing_headline(
    fault: FaultClass,
) -> None:
    result = make_case(verdicts={HEADLINE: Verdict.PASS}, fault=fault)

    assert trial_outcome(result=result, headline_check=HEADLINE) is TrialOutcome.FAIL


def test_an_upstream_fault_with_a_passing_headline_is_not_scored_a_success() -> None:
    run = make_run(make_case(verdicts={HEADLINE: Verdict.PASS}, fault=FaultClass.UPSTREAM))

    summary = summarise_run(run=run)

    assert summary.all_cases.success_rate == 0.0


@pytest.mark.parametrize("headline_check", [HEADLINE, None])
def test_a_crashed_task_fails_without_the_headline_check(headline_check: str | None) -> None:
    result = make_case(verdicts={TASK_CHECK_NAME: Verdict.FAIL})

    assert trial_outcome(result=result, headline_check=headline_check) is TrialOutcome.FAIL


@pytest.mark.parametrize("fault", [FaultClass.GRADER, FaultClass.INFRA, FaultClass.REFERENCE])
def test_an_excluding_fault_outranks_a_crashed_task(fault: FaultClass) -> None:
    result = make_case(verdicts={TASK_CHECK_NAME: Verdict.FAIL}, fault=fault)

    assert trial_outcome(result=result, headline_check=HEADLINE) is TrialOutcome.EXCLUDED


def test_the_headline_check_alone_decides_the_trial() -> None:
    passing = make_case(verdicts={HEADLINE: Verdict.PASS, "year": Verdict.FAIL})
    failing = make_case(verdicts={HEADLINE: Verdict.UNVERIFIABLE, "year": Verdict.PASS})

    assert trial_outcome(result=passing, headline_check=HEADLINE) is TrialOutcome.PASS
    assert trial_outcome(result=failing, headline_check=HEADLINE) is TrialOutcome.FAIL


def test_a_scored_trial_without_the_headline_check_raises() -> None:
    result = make_case(verdicts={"year": Verdict.PASS})

    with pytest.raises(MissingHeadlineCheckError, match="answer"):
        trial_outcome(result=result, headline_check=HEADLINE)


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        ({"a": Verdict.PASS, "b": Verdict.NOT_APPLICABLE}, TrialOutcome.PASS),
        ({"a": Verdict.PASS, "b": Verdict.FAIL}, TrialOutcome.FAIL),
        ({"a": Verdict.NOT_APPLICABLE, "b": Verdict.UNVERIFIABLE}, TrialOutcome.FAIL),
        ({}, TrialOutcome.FAIL),
    ],
)
def test_without_a_headline_check_a_trial_needs_a_pass_and_no_fail(
    verdicts: dict[str, Verdict], expected: TrialOutcome
) -> None:
    assert trial_outcome(result=make_case(verdicts=verdicts), headline_check=None) is expected


def test_a_run_is_scored_overall_and_per_tag() -> None:
    run = make_run(
        make_case(case_id="a", tags=("corpus/dac.yaml",)),
        make_case(case_id="b", verdicts={HEADLINE: Verdict.FAIL}, tags=("corpus/dac.yaml",)),
        make_case(case_id="c", tags=("corpus/imf.yaml",)),
        make_case(case_id="d", verdicts={}, fault=FaultClass.INFRA, tags=("corpus/imf.yaml",)),
    )

    summary = summarise_run(run=run)

    assert summary.all_cases == GroupSummary(
        trials_per_case=1,
        cases=4,
        scored_trials=3,
        passes=2,
        excluded_trials=1,
        cases_with_excluded_trial=1,
        fully_scored_passes=2,
        cases_passing_all=2,
        scored_outcomes=frozenset({("a", 1, True), ("b", 1, False), ("c", 1, True)}),
    )
    assert summary.all_cases.success_rate == pytest.approx(2 / 3)
    assert list(summary.by_tag) == ["corpus/dac.yaml", "corpus/imf.yaml"]
    assert summary.by_tag["corpus/dac.yaml"].success_rate == 0.5
    assert summary.by_tag["corpus/imf.yaml"].success_rate == 1.0
    assert summary.by_tag["corpus/imf.yaml"].excluded_trials == 1


def test_a_run_is_scored_by_the_headline_check_its_manifest_names() -> None:
    case = make_case(verdicts={HEADLINE: Verdict.FAIL, "year": Verdict.PASS})

    by_answer = summarise_run(run=make_run(case))
    by_year = summarise_run(run=make_run(case, headline_check="year"))
    by_every_check = summarise_run(run=make_run(case, headline_check=None))

    assert by_answer.all_cases.passes == 0
    assert by_year.all_cases.passes == 1
    assert by_every_check.all_cases.passes == 0


def test_cases_with_different_trial_counts_raise() -> None:
    """Run refuses uneven cases, so build one past validation to test the guard."""
    valid = make_run(make_case())
    results = (make_case(case_id="a", trial=1), make_case(case_id="a", trial=2), make_case())
    run = Run.model_construct(manifest=valid.manifest, results=results)

    with pytest.raises(ValueError, match=r"run 'run-1'.*without 2 trials: \['case-1'\]"):
        summarise_run(run=run)


def test_a_group_with_no_scored_trials_has_no_success_rate() -> None:
    summary = summarise_run(run=make_run(make_case(verdicts={}, fault=FaultClass.GRADER)))

    assert summary.all_cases.scored_trials == 0
    assert summary.all_cases.success_rate is None
    assert summary.all_cases.reliability_share is None


def test_a_single_trial_run_has_no_pass_distribution() -> None:
    summary = summarise_run(run=make_run(make_case()))

    assert summary.trials_per_case == 1
    assert not summary.multi_trial
    assert summary.all_cases.pass_distribution is None


def test_a_multi_trial_run_counts_fully_scored_cases_by_passing_trials() -> None:
    fail = {HEADLINE: Verdict.FAIL}
    run = make_run(
        *(make_case(case_id="all", trial=t) for t in (1, 2, 3)),
        make_case(case_id="two", trial=1),
        make_case(case_id="two", trial=2),
        make_case(case_id="two", trial=3, verdicts=fail),
        *(make_case(case_id="none", trial=t, verdicts=fail) for t in (1, 2, 3)),
    )

    summary = summarise_run(run=run)

    assert summary.multi_trial
    assert summary.all_cases.pass_distribution == (1, 0, 1, 1)
    assert summary.all_cases.cases_passing_all == 1
    assert summary.all_cases.reliability_share == pytest.approx(1 / 3)


def test_a_case_with_an_excluded_trial_leaves_the_reliability_counts() -> None:
    run = make_run(
        make_case(case_id="excl", trial=1),
        make_case(case_id="excl", trial=2),
        make_case(case_id="excl", trial=3, verdicts={}, fault=FaultClass.INFRA),
        *(make_case(case_id="all", trial=t) for t in (1, 2, 3)),
    )

    summary = summarise_run(run=run).all_cases

    assert summary.cases_with_excluded_trial == 1
    assert summary.cases_passing_all == 1
    assert summary.pass_distribution == (0, 0, 0, 1)
    assert summary.scored_trials == 5
    assert summary.passes == 5
    assert summary.fully_scored_passes == 3


def test_change_is_the_success_rate_difference_in_percentage_points() -> None:
    before = group(passes=1, fully_scored_passes=1, cases_passing_all=1)
    after = group(passes=2, fully_scored_passes=2, cases_passing_all=2)

    assert shared_change(current=after, previous=before) == SharedChange(
        change_pp=50.0, trials=2, partial=False
    )
    assert shared_change(current=before, previous=after).change_pp == -50.0


def test_change_is_taken_between_the_rates_as_displayed() -> None:
    def rate(passes: int) -> GroupSummary:
        return group(
            cases=10_000,
            scored_trials=10_000,
            passes=passes,
            fully_scored_passes=passes,
            cases_passing_all=passes,
        )

    # 33.34% shows as 33.3% and 33.36% as 33.4%, so the change must read +0.1 pp.
    assert shared_change(current=rate(3336), previous=rate(3334)).change_pp == 0.1


def test_a_group_without_a_rate_has_no_change() -> None:
    empty = group(
        scored_trials=0,
        passes=0,
        excluded_trials=2,
        cases_with_excluded_trial=2,
        fully_scored_passes=0,
        cases_passing_all=0,
    )

    assert shared_change(current=empty, previous=group()) == SharedChange(
        change_pp=None, trials=0, partial=True
    )
    assert shared_change(current=empty, previous=empty) == SharedChange(
        change_pp=None, trials=0, partial=False
    )


def test_runs_are_ordered_by_start_time_not_run_id() -> None:
    later = summarise_run(run=make_run(make_case(), run_id="a"))
    earlier = summarise_run(
        run=make_run(make_case(), run_id="b", started_at=STARTED_AT - timedelta(days=1))
    )

    assert [s.manifest.run_id for s in chronological([later, earlier])] == ["b", "a"]


def test_the_reference_groups_are_valid() -> None:
    assert group().reliability_share == 0.5
    assert multi_group().reliability_share == 0.5


def test_negative_counts_raise() -> None:
    with pytest.raises(ValueError, match="negative count"):
        group(cases_with_excluded_trial=-1)


def test_trials_not_matching_cases_times_trials_per_case_raise() -> None:
    with pytest.raises(ValueError, match="is not 2 cases of 1 trials"):
        group(excluded_trials=1)


def test_passes_above_scored_trials_raise() -> None:
    with pytest.raises(ValueError, match="exceed scored trials"):
        group(passes=3)


def test_more_cases_with_an_excluded_trial_than_excluded_trials_raise() -> None:
    with pytest.raises(ValueError, match="cases with an excluded trial"):
        group(cases_with_excluded_trial=1)


def test_fewer_scored_trials_than_fully_scored_cases_hold_raise() -> None:
    with pytest.raises(ValueError, match="fewer than the 2 fully scored cases hold"):
        group(
            scored_trials=1, excluded_trials=1, passes=0, fully_scored_passes=0, cases_passing_all=0
        )


def test_fully_scored_passes_above_passes_raise() -> None:
    with pytest.raises(ValueError, match="fully scored passes 2 exceed passes 1"):
        group(fully_scored_passes=2)


def test_more_cases_passing_all_than_fully_scored_cases_raise() -> None:
    with pytest.raises(ValueError, match="exceed fully scored cases"):
        group(scored_trials=1, excluded_trials=1, cases_with_excluded_trial=1, cases_passing_all=2)


def test_more_cases_passing_all_than_passes_allow_raise() -> None:
    with pytest.raises(ValueError, match="need more than 0 passes"):
        group(passes=0, fully_scored_passes=0)


def test_a_multi_trial_group_without_a_distribution_raises() -> None:
    with pytest.raises(ValueError, match="needs a pass distribution"):
        multi_group(pass_distribution=None)


def test_a_distribution_with_the_wrong_number_of_buckets_raises() -> None:
    with pytest.raises(ValueError, match="needs 3 buckets"):
        multi_group(pass_distribution=(0, 1, 1, 0))


def test_a_distribution_not_summing_to_the_fully_scored_cases_raises() -> None:
    with pytest.raises(ValueError, match="sums to 3, not 2 fully scored cases"):
        multi_group(pass_distribution=(1, 1, 1))


def test_a_distribution_holding_the_wrong_number_of_passes_raises() -> None:
    with pytest.raises(ValueError, match="holds 2 passes, not 3"):
        multi_group(pass_distribution=(1, 0, 1))


def test_a_top_bucket_not_matching_cases_passing_all_raises() -> None:
    with pytest.raises(ValueError, match="top bucket 1 is not cases passing all 0"):
        multi_group(cases_passing_all=0)


def test_a_group_carries_the_outcome_of_every_scored_trial() -> None:
    run = make_run(
        make_case(case_id="a", trial=1, tags=("dac",)),
        make_case(case_id="a", trial=2, verdicts={}, fault=FaultClass.INFRA, tags=("dac",)),
        make_case(case_id="b", trial=1, verdicts={HEADLINE: Verdict.FAIL}, tags=("imf",)),
        make_case(case_id="b", trial=2, tags=("imf",)),
    )

    summary = summarise_run(run=run)

    assert summary.all_cases.scored_outcomes == {("a", 1, True), ("b", 1, False), ("b", 2, True)}
    assert summary.by_tag["dac"].scored_outcomes == {("a", 1, True)}
    assert summary.by_tag["imf"].scored_outcomes == {("b", 1, False), ("b", 2, True)}


def single_trial_group(outcomes: dict[str, bool]) -> GroupSummary:
    """A single-trial group of fully scored cases with the given outcomes."""
    passes = sum(outcomes.values())
    return GroupSummary(
        trials_per_case=1,
        cases=len(outcomes),
        scored_trials=len(outcomes),
        passes=passes,
        excluded_trials=0,
        cases_with_excluded_trial=0,
        fully_scored_passes=passes,
        cases_passing_all=passes,
        scored_outcomes=frozenset((case_id, 1, ok) for case_id, ok in outcomes.items()),
    )


def test_change_is_taken_over_the_trials_both_runs_scored() -> None:
    before = single_trial_group({"a": True, "b": False, "c": False})
    after = single_trial_group({"a": True, "b": True, "d": False})

    # Over a and b: 50% before, 100% after, though the whole groups read 33.3% and 66.7%.
    assert shared_change(current=after, previous=before) == SharedChange(
        change_pp=50.0, trials=2, partial=True
    )


def test_a_change_over_a_subset_of_one_run_is_partial() -> None:
    before = single_trial_group({"a": True, "b": False})
    after = single_trial_group({"a": False})

    assert shared_change(current=after, previous=before) == SharedChange(
        change_pp=-100.0, trials=1, partial=True
    )


def test_groups_sharing_no_scored_trial_have_no_change() -> None:
    before = single_trial_group({"a": True})
    after = single_trial_group({"b": True})

    assert shared_change(current=after, previous=before) == SharedChange(
        change_pp=None, trials=0, partial=True
    )


def test_scored_outcomes_not_matching_scored_trials_raise() -> None:
    with pytest.raises(ValueError, match=r"1 scored .* pairs, not 2 scored trials"):
        group_with_outcomes(frozenset({("a", 1, True)}))


def test_scored_outcomes_not_matching_passes_raise() -> None:
    with pytest.raises(ValueError, match="0 scored outcomes passed, not 1 passes"):
        group_with_outcomes(frozenset({("a", 1, False), ("b", 1, False)}))


def test_a_scored_trial_with_two_outcomes_raises() -> None:
    with pytest.raises(ValueError, match="has two outcomes"):
        group_with_outcomes(frozenset({("a", 1, True), ("a", 1, False)}))


def group_with_outcomes(outcomes: frozenset[tuple[str, int, bool]]) -> GroupSummary:
    """The reference single-trial group of two cases, one passing, with given outcomes."""
    return GroupSummary(
        trials_per_case=1,
        cases=2,
        scored_trials=2,
        passes=1,
        excluded_trials=0,
        cases_with_excluded_trial=0,
        fully_scored_passes=1,
        cases_passing_all=1,
        scored_outcomes=outcomes,
    )


def test_a_partial_change_is_rounded_once() -> None:
    """Over three shared trials, 1 of 3 to 2 of 3 is +33.3 pp, not 66.7 - 33.3."""
    before = single_trial_group({"a": True, "b": False, "c": False, "d": True})
    after = single_trial_group({"a": True, "b": True, "c": False})

    assert shared_change(current=after, previous=before) == SharedChange(
        change_pp=33.3, trials=3, partial=True
    )
