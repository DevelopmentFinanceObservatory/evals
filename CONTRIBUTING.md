# Contributing

Issues and pull requests are welcome. The ONE Data team maintains this repository. For a large change, open an issue first so we can agree on the approach before you write it.

Please follow the [Code of Conduct](https://github.com/DevelopmentFinanceObservatory/.github/blob/main/CODE_OF_CONDUCT.md). Report security problems through the repository's private vulnerability reporting (Security tab, "Report a vulnerability"), and see the [security policy](https://github.com/DevelopmentFinanceObservatory/.github/blob/main/SECURITY.md). For questions, see [SUPPORT.md](https://github.com/DevelopmentFinanceObservatory/.github/blob/main/SUPPORT.md).

## Setup

You need [`just`](https://github.com/casey/just) and [`uv`](https://docs.astral.sh/uv/).

```bash
just setup   # uv sync, then uv run pre-commit install
```

## Before you open a pull request

```bash
just check    # lint, type check, test: CI's lint and test jobs
just format   # apply ruff fixes and formatting
```

CI runs `lint` and `test` on every ready pull request, with `test` on Python 3.12 and 3.13. `just check` covers both on your local Python. CI skips draft pull requests.

Pre-commit runs ruff, `ty`, codespell, actionlint, and mdformat. Ruff and `ty` settings are in `pyproject.toml`.

## Branches and commits

`dev` is the trunk. Branch from it and open pull requests against it. `main` is the release branch: a pull request from `dev` to `main` needs one approval and merges with a merge commit. Commit subjects follow Conventional Commits (`feat:`, `fix:`, `docs:`, `ci:`, `build:`).

## What a pull request needs

- `just check` passes, and CI is green.
- New behaviour has tests, and a bug fix has a test that fails without it.
- Docs change with the code.
- A change that needs a matching change elsewhere, such as a component suite's pinned revision or a new secret, says so in the pull request.
