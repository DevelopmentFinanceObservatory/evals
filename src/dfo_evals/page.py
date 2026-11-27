"""Build one static HTML page showing how each suite is doing over time.

Run as ``python -m dfo_evals.page --root runs --out site/index.html``. Each run is
scored by the headline check its manifest names. The page holds no generated-at time,
so the same runs always build the same bytes.
"""

import argparse
import logging
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Final

from dfo_evals.records import RecordSchemaError, iter_runs, read_run
from dfo_evals.summary import (
    PERCENT_DECIMALS,
    GroupSummary,
    RunSummary,
    chronological,
    display_percent,
    rate_change_pp,
    summarise_run,
)

logger = logging.getLogger(__name__)

PAGE_TITLE: Final = "DFO eval results"
HISTORY_LIMIT: Final = 30
ALL_CASES_LABEL: Final = "All cases"
NO_VALUE: Final = "n/a"
# A change cell with nothing to compare against: a new group or the suite's first run.
NO_PREVIOUS: Final = "new"
# A change cell whose previous run scored a different set of cases (another dataset
# digest), so a change in rate would not measure the system.
CORPUS_CHANGED: Final = "corpus changed"
# A change cell whose previous run named another headline check, so its trials passed
# by another rule.
HEADLINE_CHANGED: Final = "headline changed"
UNREADABLE_HEADING: Final = "Unreadable runs"
MINUS_SIGN: Final = "\N{MINUS SIGN}"
RANGE_SEPARATOR: Final = "\N{EN DASH}"
DISTRIBUTION_SEPARATOR: Final = " \N{MIDDLE DOT} "
# Run, start time and revision are text; the rest of the history columns are numbers.
_HISTORY_TEXT_COLUMNS: Final = 3
_NUM_CLASS: Final = ' class="num"'

_CSS: Final = """
:root {
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #5c5c5c;
  --rule: #d6d6d6;
  --head-bg: #f2f2f2;
  --up: #1b6e3a;
  --down: #a3261b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #121212;
    --fg: #ececec;
    --muted: #a8a8a8;
    --rule: #3a3a3a;
    --head-bg: #1f1f1f;
    --up: #6fd08f;
    --down: #ff8a80;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0 auto;
  max-width: 64rem;
  padding: 1rem;
  background: var(--bg);
  color: var(--fg);
  font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
h1 { font-size: 1.5rem; margin: 0 0 1rem; }
h2 { font-size: 1.25rem; margin: 2rem 0 0.5rem; }
p { margin: 0.25rem 0; }
.meta, .note, footer { color: var(--muted); font-size: 0.875rem; }
.table-wrap { overflow-x: auto; margin: 0.75rem 0; }
table { border-collapse: collapse; min-width: 100%; }
caption { text-align: left; font-weight: 600; padding: 0.25rem 0; }
th, td { padding: 0.375rem 0.625rem; border-bottom: 1px solid var(--rule); text-align: left;
  white-space: nowrap; }
thead th { background: var(--head-bg); }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.up { color: var(--up); }
.down { color: var(--down); }
code { font-size: 0.9em; }
footer { margin-top: 2rem; border-top: 1px solid var(--rule); padding-top: 0.5rem; }
"""


def build_page(*, root: Path) -> str:
    """Build the results page from every run under a root.

    Args:
        root: Directory holding ``<suite>/<run_id>/`` run records.

    A run that fails to read, such as one of another schema version or one missing its
    results, is left out and listed at the end of the page, so one bad run leaves every
    other run on the page. The log names it by suite and run directory only, since a
    validation error can quote record contents.

    Returns:
        One self-contained HTML document.

    Raises:
        ValueError: The root holds no readable runs.
        FileNotFoundError: The root does not exist.
    """
    by_suite: defaultdict[str, list[RunSummary]] = defaultdict(list)
    unreadable: list[str] = []
    for run_dir in iter_runs(root=root):
        try:
            run = read_run(run_dir)
        except (RecordSchemaError, FileNotFoundError) as error:
            name = f"{run_dir.parent.name}/{run_dir.name}"
            logger.warning("left out unreadable run %s (%s)", name, type(error).__name__)
            unreadable.append(name)
            continue
        by_suite[run.manifest.suite].append(summarise_run(run=run))
    if not by_suite:
        raise ValueError(f"no readable runs under {root}; unreadable: {unreadable}")
    logger.info(
        "read %d runs across %d suites",
        sum(len(runs) for runs in by_suite.values()),
        len(by_suite),
    )

    sections = [
        _suite_section(suite=suite, runs=chronological(by_suite[suite]))
        for suite in sorted(by_suite)
    ]
    if unreadable:
        sections.append(_unreadable_section(unreadable))
    return _document("\n".join(sections))


def _unreadable_section(names: Sequence[str]) -> str:
    """List the runs left out because they failed to read."""
    items = "".join(f"<li><code>{escape(name)}</code></li>" for name in names)
    return (
        "<section>"
        f"<h2>{escape(UNREADABLE_HEADING)}</h2>"
        '<p class="note">These runs failed to read with this version of dfo-evals and are '
        "left out of the tables above.</p>"
        f"<ul>{items}</ul>"
        "</section>"
    )


def format_rate(rate: float | None) -> str:
    """Format a share as a percentage with one decimal, or "n/a" when None."""
    percent = display_percent(rate)
    if percent is None:
        return NO_VALUE
    return f"{percent:.{PERCENT_DECIMALS}f}%"


def format_change(change_pp: float | None) -> str:
    """Format a change in percentage points with an explicit sign and a real minus.

    Returns "n/a" when the change is None because a rate is missing.
    """
    if change_pp is None:
        return NO_VALUE
    rounded = round(change_pp, PERCENT_DECIMALS)
    magnitude = f"{abs(rounded):.{PERCENT_DECIMALS}f} pp"
    if rounded > 0:
        return f"+{magnitude}"
    if rounded < 0:
        return f"{MINUS_SIGN}{magnitude}"
    return magnitude


def format_time(moment: datetime) -> str:
    """Format a timestamp as ISO 8601 in UTC, to the minute."""
    return moment.astimezone(UTC).isoformat(timespec="minutes")


def _suite_section(*, suite: str, runs: Sequence[RunSummary]) -> str:
    """Render one suite: latest run, group table and history."""
    latest = runs[-1]
    previous = runs[-2] if len(runs) > 1 else None
    manifest = latest.manifest
    headline_check = manifest.headline_check
    scoring = (
        f"the headline check <code>{escape(headline_check)}</code> passes"
        if headline_check is not None
        else "no check fails and at least one passes"
    )
    return "\n".join(
        (
            f'<section aria-labelledby="{_suite_id(suite)}">',
            f'<h2 id="{_suite_id(suite)}">{escape(suite)}</h2>',
            '<p class="meta">Latest run <code>'
            f"{escape(manifest.run_id)}</code>, started {escape(format_time(manifest.started_at))}"
            f", revision <code>{escape(manifest.revision)}</code>.</p>",
            f'<p class="note">In the latest run a trial passes when {scoring}. Excluded trials are grader, '
            "infra or reference faults and do not count towards scores.</p>",
            _groups_table(suite=suite, latest=latest, previous=previous),
            _history_table(suite=suite, runs=runs),
            "</section>",
        )
    )


def _groups_table(*, suite: str, latest: RunSummary, previous: RunSummary | None) -> str:
    """Render the latest run's scores per group, with change since the previous run.

    Per-case columns appear for multi-trial suites only; with one trial per case they
    repeat the success rate.
    """
    headers = ["Group", "Scored trials", "Success rate", "Change since previous run"]
    if latest.multi_trial:
        headers += [
            "Cases passing every trial",
            f"Cases by passing trials (0{RANGE_SEPARATOR}{latest.trials_per_case})",
        ]
    headers.append("Excluded trials")

    incomparable = _incomparable(latest, previous)
    rows: list[tuple[str, GroupSummary, GroupSummary | None]] = [
        (ALL_CASES_LABEL, latest.all_cases, previous.all_cases if previous else None)
    ]
    rows.extend(
        (tag, group, previous.by_tag.get(tag) if previous else None)
        for tag, group in latest.by_tag.items()
    )

    body = []
    for label, group, before in rows:
        cells = [
            _num(str(group.scored_trials)),
            _num(format_rate(group.success_rate)),
            _change_cell(current=group, previous=before, incomparable=incomparable),
        ]
        if latest.multi_trial:
            cells += [
                _num(f"{group.cases_passing_all} of {group.cases_fully_scored}"),
                _num(_format_distribution(group.pass_distribution)),
                _num(_format_excluded(group)),
            ]
        else:
            cells.append(_num(str(group.excluded_trials)))
        body.append(f'<tr><th scope="row">{escape(label)}</th>{"".join(cells)}</tr>')
    return _table(caption=f"Latest run of {suite} by group", headers=headers, rows=body)


def _history_table(*, suite: str, runs: Sequence[RunSummary]) -> str:
    """Render the last runs, newest first, with all-cases success rate and change."""
    previous_runs: list[RunSummary | None] = [None, *runs[:-1]]
    recent = list(zip(runs, previous_runs, strict=True))[-HISTORY_LIMIT:]
    body = [
        f'<tr><th scope="row"><code>{escape(run.manifest.run_id)}</code></th>'
        f"<td>{escape(format_time(run.manifest.started_at))}</td>"
        f"<td><code>{escape(run.manifest.revision)}</code></td>"
        f"{_num(format_rate(run.all_cases.success_rate))}"
        + _change_cell(
            current=run.all_cases,
            previous=before.all_cases if before else None,
            incomparable=_incomparable(run, before),
        )
        + "</tr>"
        for run, before in reversed(recent)
    ]
    shown = f"last {len(recent)} runs" if len(recent) > 1 else "1 run"
    return _table(
        caption=f"History of {suite}, {shown}, newest first",
        headers=["Run", "Started (UTC)", "Revision", "Success rate", "Change"],
        rows=body,
        text_columns=_HISTORY_TEXT_COLUMNS,
    )


def _table(
    *, caption: str, headers: Sequence[str], rows: Sequence[str], text_columns: int = 1
) -> str:
    """Render a table in a scrolling wrapper; headers after ``text_columns`` are numeric."""
    head = "".join(
        f'<th scope="col"{_NUM_CLASS if index >= text_columns else ""}>{escape(header)}</th>'
        for index, header in enumerate(headers)
    )
    return (
        f'<div class="table-wrap" role="region" tabindex="0" aria-label="{escape(caption)}">'
        f"<table><caption>{escape(caption)}</caption>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _num(text: str) -> str:
    """Render a right-aligned numeric cell."""
    return f'<td class="num">{escape(text)}</td>'


def _incomparable(run: RunSummary, previous: RunSummary | None) -> str | None:
    """Say why a run's rates cannot be compared with the run before it, or None.

    Runs on different corpora (dataset digests) or scored by different headline checks
    differ in rate without any change in the system.
    """
    if previous is None:
        return None
    if run.manifest.dataset_digest != previous.manifest.dataset_digest:
        return CORPUS_CHANGED
    if run.manifest.headline_check != previous.manifest.headline_check:
        return HEADLINE_CHANGED
    return None


def _change_cell(
    *, current: GroupSummary, previous: GroupSummary | None, incomparable: str | None
) -> str:
    """Render a change cell; colour backs up the sign, never replaces it.

    A change against a run it cannot be compared with shows the reason, uncoloured.
    """
    if previous is None:
        return _num(NO_PREVIOUS)
    if incomparable is not None:
        return _num(incomparable)
    text = format_change(rate_change_pp(current=current, previous=previous))
    if text.startswith("+"):
        return f'<td class="num up">{escape(text)}</td>'
    if text.startswith(MINUS_SIGN):
        return f'<td class="num down">{escape(text)}</td>'
    return _num(text)


def _format_excluded(group: GroupSummary) -> str:
    """Render excluded trials with the cases they fall in, as ``k in n cases``."""
    noun = "case" if group.cases_with_excluded_trial == 1 else "cases"
    return f"{group.excluded_trials} in {group.cases_with_excluded_trial} {noun}"


def _format_distribution(distribution: tuple[int, ...] | None) -> str:
    """Render a pass-count distribution as ``0: a · 1: b · ...``."""
    if distribution is None:
        return NO_VALUE
    return DISTRIBUTION_SEPARATOR.join(
        f"{passes}: {cases}" for passes, cases in enumerate(distribution)
    )


def _suite_id(suite: str) -> str:
    """Return the element id for a suite heading; suite names are safe directory names."""
    return escape(f"suite-{suite}")


def _document(body: str) -> str:
    """Wrap section markup in a complete HTML document."""
    return "\n".join(
        (
            "<!DOCTYPE html>",
            '<html lang="en-GB">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(PAGE_TITLE)}</title>",
            f"<style>{_CSS}</style>",
            "</head>",
            "<body>",
            "<main>",
            f"<h1>{escape(PAGE_TITLE)}</h1>",
            body,
            "</main>",
            "<footer><p>Success rate is passes over scored trials. Change is in percentage "
            "points against the previous run of the same suite, shown only when both runs scored the "
            "same corpus by the same headline check. Tags can overlap, so group rows need not "
            "add up to All cases.</p></footer>",
            "</body>",
            "</html>",
            "",
        )
    )


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="dfo-evals-page", description="Build the DFO eval results page."
    )
    parser.add_argument("--root", type=Path, required=True, help="directory of run records")
    parser.add_argument("--out", type=Path, required=True, help="HTML file to write")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build the page and write it to ``--out``.

    Args:
        argv: Command-line arguments, or None for ``sys.argv``.

    Returns:
        Process exit code, 0 on success.

    Raises:
        ValueError: The root holds no runs.
        FileNotFoundError: The root does not exist.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args(argv)
    page = build_page(root=args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(page)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
