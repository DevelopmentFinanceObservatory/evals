# DFO evals

The shared eval library of the [Development Finance Observatory](https://github.com/DevelopmentFinanceObservatory) (DFO) and the home of its outcome suites. Component suites in other DFO repositories, such as [`apis`](https://github.com/DevelopmentFinanceObservatory/apis) and [`ai`](https://github.com/DevelopmentFinanceObservatory/ai), depend on it by git revision. The DFO is an initiative of [ONE Data](https://data.one.org), built on [Google Data Commons](https://datacommons.org). The organisation profile has the full map of repositories.

This repository is under active development. Interfaces, schemas, and data may change without notice.

## What's inside

```text
src/dfo_evals/   The `dfo-evals` Python package
tests/           The package's test suite
```

The package supports Python 3.12 and later. `apis` runs its suites on 3.12 and `ai` on 3.13, and CI tests both.

`dfo_evals.langfuse_sink`, which pushes runs to Langfuse, needs the `langfuse` extra: depend on `dfo-evals[langfuse]`.

## Getting started

You need [`just`](https://github.com/casey/just) and [`uv`](https://docs.astral.sh/uv/).

```bash
just setup   # sync dependencies and install pre-commit hooks
just check   # lint, type check, and tests, as CI runs them
```

`just` lists every recipe.

## Eval results

The `Publish eval results` workflow on `dev` builds the report daily from the complete runs in the dev bucket. Dispatch the workflow from `dev` to refresh it on demand.

Open the [eval results page](https://storage.cloud.google.com/dfo-dev-eval-runs-dev-fin-observatory/site/index.html) with a Google account that has access to the bucket. The [infra runbook](https://github.com/DevelopmentFinanceObservatory/infra/blob/dev/infra/dfo-dev/apis/README.md#eval-runs) describes how to grant access.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

MIT. See [LICENSE](LICENSE).
