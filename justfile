# dfo-evals developer commands
# Run 'just' or 'just --list' to see all available targets.

# Show available commands (default)
_default:
    @just --list --unsorted

# First-time setup: sync dependencies and install pre-commit hooks
setup: sync
    uv run pre-commit install

# Install the package and dev dependencies into .venv
sync:
    uv sync

# Lint and format check, without fixes
lint:
    uv run ruff check .
    uv run ruff format --check .

# Apply ruff fixes and formatting
format:
    uv run ruff check --fix .
    uv run ruff format .

# Type check with ty (settings in pyproject.toml)
typecheck:
    uv run ty check

# Run the test suite
test:
    uv run pytest

# Everything CI's lint and test jobs run. Green predicts green
check: lint typecheck test
