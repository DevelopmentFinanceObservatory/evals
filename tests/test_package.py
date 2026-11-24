"""Package smoke test."""

from importlib.metadata import version

import dfo_evals


def test_package_exposes_installed_version() -> None:
    assert dfo_evals.__version__ == version("dfo-evals")
