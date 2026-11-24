# dfo-evals

`dfo-evals` is the shared eval library for the Development Finance Observatory (DFO) and the home of its outcome suites. Component suites in other DFO repositories depend on it by git revision.

## Development

You need [`just`](https://github.com/casey/just) and [`uv`](https://docs.astral.sh/uv/). The package supports Python 3.12 and later.

```bash
just sync    # install the package and dev dependencies
just check   # ruff, ty and pytest, as CI runs them
```

`just fmt` applies ruff fixes and formatting. See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.
