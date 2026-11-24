"""Score run records: per-trial outcomes, per-group rates and change between runs.

Scoring rules:

- A trial whose fault is grader, infra or reference is excluded: setup or grading failed,
  not the system under test.
- Otherwise a trial passes when its headline verdict is PASS and fails otherwise, so
  upstream and agent faults count as fails. The headline verdict is the named headline
  check's verdict; with no headline check, a trial passes when no check is FAIL and at
  least one check is PASS.
- A case with any excluded trial is left out of the per-case reliability counts, since
  its remaining trials cannot show whether it passes every time.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from dfo_evals.records import CaseResult, FaultClass, Run, RunManifest, Verdict

EXCLUDED_FAULTS: Final = frozenset({FaultClass.GRADER, FaultClass.INFRA, FaultClass.REFERENCE})
PERCENT: Final = 100.0
PERCENT_DECIMALS: Final = 1
_SINGLE_TRIAL: Final = 1


class TrialOutcome(StrEnum):
    """How one trial counts towards scores."""

    PASS = "pass"
    FAIL = "fail"
    EXCLUDED = "excluded"


class MissingHeadlineCheckError(ValueError):
    """A scored trial lacks the suite's headline check."""


@dataclass(frozen=True)
class GroupSummary:
    """Scores for one group of trials in one run.

    A case is fully scored when none of its trials is excluded.

    Attributes:
        trials_per_case: Trials every case in the run has.
        cases: Distinct cases in the group.
        scored_trials: Trials that are not excluded.
        passes: Scored trials that passed.
        excluded_trials: Trials excluded by a grader, infra or reference fault.
        cases_with_excluded_trial: Cases with at least one excluded trial.
        fully_scored_passes: Passing trials of fully scored cases.
        cases_passing_all: Fully scored cases whose trials all pass.
        pass_distribution: Count of fully scored cases by number of passing trials,
            index 0 to ``trials_per_case``; None for single-trial runs.
    """

    trials_per_case: int
    cases: int
    scored_trials: int
    passes: int
    excluded_trials: int
    cases_with_excluded_trial: int
    fully_scored_passes: int
    cases_passing_all: int
    pass_distribution: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Reject counts that cannot come from real trials."""
        self._check_counts()
        self._check_rates()
        self._check_distribution()

    @property
    def cases_fully_scored(self) -> int:
        """Cases with no excluded trial."""
        return self.cases - self.cases_with_excluded_trial

    @property
    def success_rate(self) -> float | None:
        """Passes over scored trials, or None with no scored trials."""
        return _share(self.passes, self.scored_trials)

    @property
    def reliability_share(self) -> float | None:
        """Share of fully scored cases passing every trial, or None with none."""
        return _share(self.cases_passing_all, self.cases_fully_scored)

    def _check_counts(self) -> None:
        """Check counts are non-negative and consistent with each other."""
        if self.trials_per_case < _SINGLE_TRIAL:
            raise ValueError(f"trials_per_case must be at least 1, got {self.trials_per_case}")
        counts = (
            self.cases,
            self.scored_trials,
            self.passes,
            self.excluded_trials,
            self.cases_with_excluded_trial,
            self.fully_scored_passes,
            self.cases_passing_all,
        )
        if any(count < 0 for count in counts):
            raise ValueError(f"negative count in {self}")
        if self.scored_trials + self.excluded_trials != self.cases * self.trials_per_case:
            raise ValueError(
                f"scored {self.scored_trials} plus excluded {self.excluded_trials} trials is "
                f"not {self.cases} cases of {self.trials_per_case} trials"
            )
        if self.passes > self.scored_trials:
            raise ValueError(f"passes {self.passes} exceed scored trials {self.scored_trials}")
        if self.cases_with_excluded_trial > min(self.cases, self.excluded_trials):
            raise ValueError(
                f"{self.cases_with_excluded_trial} cases with an excluded trial, but only "
                f"{self.cases} cases and {self.excluded_trials} excluded trials"
            )
        if self.scored_trials < self.cases_fully_scored * self.trials_per_case:
            raise ValueError(
                f"scored trials {self.scored_trials} fewer than the "
                f"{self.cases_fully_scored} fully scored cases hold"
            )
        if self.fully_scored_passes > self.passes:
            raise ValueError(
                f"fully scored passes {self.fully_scored_passes} exceed passes {self.passes}"
            )
        if self.cases_passing_all > self.cases_fully_scored:
            raise ValueError(
                f"cases passing all {self.cases_passing_all} exceed fully scored cases "
                f"{self.cases_fully_scored}"
            )
        if self.cases_passing_all * self.trials_per_case > self.fully_scored_passes:
            raise ValueError(
                f"{self.cases_passing_all} cases passing all {self.trials_per_case} trials "
                f"need more than {self.fully_scored_passes} passes"
            )

    def _check_rates(self) -> None:
        """Check rates lie within [0, 1]."""
        for name, rate in (
            ("success rate", self.success_rate),
            ("reliability share", self.reliability_share),
        ):
            if rate is not None and not 0.0 <= rate <= 1.0:
                raise ValueError(f"{name} {rate} outside [0, 1]")

    def _check_distribution(self) -> None:
        """Check the pass distribution agrees with the case and pass counts."""
        distribution = self.pass_distribution
        if distribution is None:
            if self.trials_per_case > _SINGLE_TRIAL:
                raise ValueError("a multi-trial group needs a pass distribution")
            return
        if len(distribution) != self.trials_per_case + 1:
            raise ValueError(
                f"pass distribution {distribution} needs {self.trials_per_case + 1} buckets"
            )
        if any(count < 0 for count in distribution):
            raise ValueError(f"negative count in pass distribution {distribution}")
        if sum(distribution) != self.cases_fully_scored:
            raise ValueError(
                f"pass distribution {distribution} sums to {sum(distribution)}, "
                f"not {self.cases_fully_scored} fully scored cases"
            )
        weighted = sum(passes * cases for passes, cases in enumerate(distribution))
        if weighted != self.fully_scored_passes:
            raise ValueError(
                f"pass distribution {distribution} holds {weighted} passes, "
                f"not {self.fully_scored_passes}"
            )
        if distribution[-1] != self.cases_passing_all:
            raise ValueError(
                f"top bucket {distribution[-1]} is not cases passing all {self.cases_passing_all}"
            )


@dataclass(frozen=True)
class RunSummary:
    """Scores for one run: every case together and each tag on its own.

    Attributes:
        manifest: The run's manifest.
        trials_per_case: Trials every case in the run has.
        all_cases: Scores over every trial in the run.
        by_tag: Scores per tag, keyed by tag in sorted order.
    """

    manifest: RunManifest
    trials_per_case: int
    all_cases: GroupSummary
    by_tag: dict[str, GroupSummary]

    @property
    def multi_trial(self) -> bool:
        """Whether each case runs more than once."""
        return self.trials_per_case > _SINGLE_TRIAL


def headline_passes(*, result: CaseResult, headline_check: str | None) -> bool:
    """Decide whether a trial's headline verdict is a pass.

    Args:
        result: The trial.
        headline_check: Name of the check that decides the trial, or None to require
            no FAIL and at least one PASS across all checks.

    Returns:
        True when the headline verdict is PASS.

    Raises:
        MissingHeadlineCheckError: The trial has no check named ``headline_check``.
    """
    if headline_check is None:
        verdicts = {check.verdict for check in result.checks}
        return Verdict.FAIL not in verdicts and Verdict.PASS in verdicts
    for check in result.checks:
        if check.name == headline_check:
            return check.verdict is Verdict.PASS
    raise MissingHeadlineCheckError(
        f"case {result.case_id!r} trial {result.trial} has no headline check {headline_check!r}"
    )


def trial_outcome(*, result: CaseResult, headline_check: str | None) -> TrialOutcome:
    """Classify one trial as excluded, passing or failing.

    Args:
        result: The trial.
        headline_check: Name of the suite's headline check, or None.

    Returns:
        EXCLUDED for grader, infra or reference faults; otherwise PASS or FAIL by the
        headline verdict.

    Raises:
        MissingHeadlineCheckError: A scored trial has no headline check.
    """
    if result.fault in EXCLUDED_FAULTS:
        return TrialOutcome.EXCLUDED
    if headline_passes(result=result, headline_check=headline_check):
        return TrialOutcome.PASS
    return TrialOutcome.FAIL


def summarise_group(
    *, outcomes: Sequence[tuple[str, TrialOutcome]], trials_per_case: int
) -> GroupSummary:
    """Score one group of trials.

    Args:
        outcomes: One ``(case_id, outcome)`` pair per trial in the group.
        trials_per_case: Trials every case in the run has; above one, the summary
            carries the distribution of cases by passing trials.

    Returns:
        The group's scores.

    Raises:
        ValueError: A case does not have ``trials_per_case`` trials.
    """
    by_case: defaultdict[str, Counter[TrialOutcome]] = defaultdict(Counter)
    for case_id, outcome in outcomes:
        by_case[case_id][outcome] += 1
    uneven = sorted(case for case, counts in by_case.items() if counts.total() != trials_per_case)
    if uneven:
        raise ValueError(f"cases without {trials_per_case} trials: {uneven}")

    fully_scored = [counts for counts in by_case.values() if not counts[TrialOutcome.EXCLUDED]]
    distribution = None
    if trials_per_case > _SINGLE_TRIAL:
        passes_per_case = Counter(counts[TrialOutcome.PASS] for counts in fully_scored)
        distribution = tuple(passes_per_case[count] for count in range(trials_per_case + 1))

    totals = Counter(outcome for _, outcome in outcomes)
    return GroupSummary(
        trials_per_case=trials_per_case,
        cases=len(by_case),
        scored_trials=totals[TrialOutcome.PASS] + totals[TrialOutcome.FAIL],
        passes=totals[TrialOutcome.PASS],
        excluded_trials=totals[TrialOutcome.EXCLUDED],
        cases_with_excluded_trial=len(by_case) - len(fully_scored),
        fully_scored_passes=sum(counts[TrialOutcome.PASS] for counts in fully_scored),
        cases_passing_all=sum(1 for counts in fully_scored if not counts[TrialOutcome.FAIL]),
        pass_distribution=distribution,
    )


def summarise_run(*, run: Run, headline_check: str | None) -> RunSummary:
    """Score a run as a whole and per tag.

    Args:
        run: The run to score.
        headline_check: Name of the suite's headline check, or None.

    Returns:
        The run's scores.

    Raises:
        MissingHeadlineCheckError: A scored trial has no headline check.
        ValueError: Cases in the run have different numbers of trials.
    """
    where = f"suite {run.manifest.suite!r} run {run.manifest.run_id!r}"
    trials_per_case = max(
        Counter(result.case_id for result in run.results).values(), default=_SINGLE_TRIAL
    )

    all_outcomes: list[tuple[str, TrialOutcome]] = []
    tag_outcomes: defaultdict[str, list[tuple[str, TrialOutcome]]] = defaultdict(list)
    for result in run.results:
        try:
            outcome = trial_outcome(result=result, headline_check=headline_check)
        except MissingHeadlineCheckError as error:
            raise MissingHeadlineCheckError(f"{where}: {error}") from error
        pair = (result.case_id, outcome)
        all_outcomes.append(pair)
        for tag in dict.fromkeys(result.tags):
            tag_outcomes[tag].append(pair)

    # Every tag group's cases are a subset of all cases, so checking here covers them.
    try:
        all_cases = summarise_group(outcomes=all_outcomes, trials_per_case=trials_per_case)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error
    return RunSummary(
        manifest=run.manifest,
        trials_per_case=trials_per_case,
        all_cases=all_cases,
        by_tag={
            tag: summarise_group(outcomes=tag_outcomes[tag], trials_per_case=trials_per_case)
            for tag in sorted(tag_outcomes)
        },
    )


def display_percent(rate: float | None) -> float | None:
    """Convert a share to a percentage rounded as the page shows it, or None."""
    return None if rate is None else round(rate * PERCENT, PERCENT_DECIMALS)


def rate_change_pp(*, current: GroupSummary, previous: GroupSummary) -> float | None:
    """Change in success rate between two runs, in percentage points.

    The change is taken between the rates as displayed, so it always agrees with them.

    Args:
        current: The group in the later run.
        previous: The same group in the earlier run.

    Returns:
        The displayed current rate minus the displayed previous rate, or None when
        either rate is missing.
    """
    now = display_percent(current.success_rate)
    before = display_percent(previous.success_rate)
    if now is None or before is None:
        return None
    return round(now - before, PERCENT_DECIMALS)


def chronological(summaries: Iterable[RunSummary]) -> list[RunSummary]:
    """Order run summaries oldest first by start time, then run id.

    Args:
        summaries: Summaries of one suite's runs.

    Returns:
        The summaries in run order.

    Raises:
        ValueError: The summaries span more than one suite.
    """
    ordered = sorted(summaries, key=lambda s: (s.manifest.started_at, s.manifest.run_id))
    suites = {summary.manifest.suite for summary in ordered}
    if len(suites) > 1:
        raise ValueError(f"summaries span several suites: {sorted(suites)}")
    return ordered


def _share(part: int, whole: int) -> float | None:
    """Return ``part / whole``, or None when ``whole`` is zero."""
    return part / whole if whole else None
