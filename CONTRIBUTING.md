# Contributing

## Setup

```bash
just sync
uv run pre-commit install
```

## Workflow

- `main` is the trunk. Branch from it and open pull requests against it.
- Commit subjects follow Conventional Commits (`feat:`, `fix:`, `docs:`, `ci:`, `build:`).
- Pull requests stack and each does one thing.
- `just check` passes before you ask for review. CI runs the same lint and test steps and skips draft pull requests.
