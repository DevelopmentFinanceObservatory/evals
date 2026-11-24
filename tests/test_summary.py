"""Tests for scoring run records."""

from datetime import UTC, datetime, timedelta

import pytest

from dfo_evals.records import CaseResult, CheckResult, FaultClass, Run, RunManifest, Verdict
from dfo_evals.summary import (
    GroupSummary,
    MissingHeadlineCheckError,
    TrialOutcome,
    chronological,
    rate_change_pp,
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


def make_run(*results: CaseResult, run_id: str = "run-1", started_at: datetime = STARTED_AT) -> Run:
    manifest = RunManifest(
        suite="apis-search",
        run_id=run_id,
        started_at=started_at,
        finished_at=None,
        revision="abc123",
        environment={},
        dataset_digest="sha256:dataset",
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
    return GroupSummary(**fields)


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
    return GroupSummary(**fields)


@pytest.mark.parametrize("fault", [FaultClass.GRADER, FaultClass.INFRA, FaultClass.REFERENCE])
def test_grader_infra_and_reference_faults_exclude_the_trial(fault: FaultClass) -> None:
    result = make_case(verdicts={}, fault=fault)

    assert trial_outcome(result=result, headline_check=HEADLINE) is TrialOutcome.EXCLUDED


@pytest.mark.parametrize("fault", [FaultClass.AGENT, FaultClass.UPSTREAM])
def test_agent_and_upstream_faults_count_as_fails(fault: FaultClass) -> None:
    result = make_case(verdicts={HEADLINE: Verdict.FAIL}, fault=fault)

    assert trial_outcome(result=result, headline_check=HEADLINE) is TrialOutcome.FAIL


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

    summary = summarise_run(run=run, headline_check=HEADLINE)

    assert summary.all_cases == GroupSummary(
        trials_per_case=1,
        cases=4,
        scored_trials=3,
        passes=2,
        excluded_trials=1,
        cases_with_excluded_trial=1,
        fully_scored_passes=2,
        cases_passing_all=2,
    )
    assert summary.all_cases.success_rate == pytest.approx(2 / 3)
    assert list(summary.by_tag) == ["corpus/dac.yaml", "corpus/imf.yaml"]
    assert summary.by_tag["corpus/dac.yaml"].success_rate == 0.5
    assert summary.by_tag["corpus/imf.yaml"].success_rate == 1.0
    assert summary.by_tag["corpus/imf.yaml"].excluded_trials == 1


def test_a_missing_headline_check_in_a_run_names_the_suite_and_run() -> None:
    run = make_run(make_case(verdicts={"year": Verdict.PASS}), run_id="night-7")

    with pytest.raises(MissingHeadlineCheckError, match=r"'apis-search' run 'night-7'.*answer"):
        summarise_run(run=run, headline_check=HEADLINE)


def test_cases_with_different_trial_counts_raise() -> None:
    run = make_run(make_case(case_id="a", trial=1), make_case(case_id="a", trial=2), make_case())

    with pytest.raises(ValueError, match=r"run 'run-1'.*without 2 trials: \['case-1'\]"):
        summarise_run(run=run, headline_check=HEADLINE)


def test_a_group_with_no_scored_trials_has_no_success_rate() -> None:
    summary = summarise_run(
        run=make_run(make_case(verdicts={}, fault=FaultClass.GRADER)), headline_check=HEADLINE
    )

    assert summary.all_cases.scored_trials == 0
    assert summary.all_cases.success_rate is None
    assert summary.all_cases.reliability_share is None


def test_a_single_trial_run_has_no_pass_distribution() -> None:
    summary = summarise_run(run=make_run(make_case()), headline_check=HEADLINE)

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

    summary = summarise_run(run=run, headline_check=HEADLINE)

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

    summary = summarise_run(run=run, headline_check=HEADLINE).all_cases

    assert summary.cases_with_excluded_trial == 1
    assert summary.cases_passing_all == 1
    assert summary.pass_distribution == (0, 0, 0, 1)
    assert summary.scored_trials == 5
    assert summary.passes == 5
    assert summary.fully_scored_passes == 3


def test_change_is_the_success_rate_difference_in_percentage_points() -> None:
    before = group(passes=1, fully_scored_passes=1, cases_passing_all=1)
    after = group(passes=2, fully_scored_passes=2, cases_passing_all=2)

    assert rate_change_pp(current=after, previous=before) == 50.0
    assert rate_change_pp(current=before, previous=after) == -50.0


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
    assert rate_change_pp(current=rate(3336), previous=rate(3334)) == 0.1


def test_a_group_without_a_rate_has_no_change() -> None:
    empty = group(
        scored_trials=0,
        passes=0,
        excluded_trials=2,
        cases_with_excluded_trial=2,
        fully_scored_passes=0,
        cases_passing_all=0,
    )

    assert rate_change_pp(current=empty, previous=group()) is None


def test_runs_are_ordered_by_start_time_not_run_id() -> None:
    later = summarise_run(run=make_run(make_case(), run_id="a"), headline_check=HEADLINE)
    earlier = summarise_run(
        run=make_run(make_case(), run_id="b", started_at=STARTED_AT - timedelta(days=1)),
        headline_check=HEADLINE,
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
