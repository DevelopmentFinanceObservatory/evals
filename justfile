# dfo-evals developer commands
# Run 'just' or 'just --list' to see all available targets.

# Show available commands (default)
_default:
    @just --list --unsorted

# Install the package and dev dependencies into .venv
sync:
    uv sync

# Lint (no fixes) + format check + type check
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run ty check

# Apply ruff fixes and formatting
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# Run the test suite
test:
    uv run pytest

# Everything CI's lint and test jobs run. Green predicts green
check: lint test
