"""Publish the results page from the complete run records in Cloud Storage."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from google.api_core.exceptions import GoogleAPIError, NotFound
    from google.auth.exceptions import GoogleAuthError
    from google.cloud import storage
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        f"dfo_evals.publish needs the storage extra (dfo-evals[storage]): {error}"
    ) from error
from pydantic import TypeAdapter, ValidationError

from dfo_evals.page import build_page
from dfo_evals.records import MANIFEST_FILE, RESERVED_SUITE, RESULTS_FILE, SafeName

logger = logging.getLogger(__name__)
_PAGE_OBJECT = "site/index.html"
_SAFE_NAME = TypeAdapter(SafeName)


def _run_path(name: str) -> tuple[str, str] | None:
    parts = name.split("/")
    if len(parts) != 3 or parts[-1] != MANIFEST_FILE:
        return None
    suite, run_id, _ = parts
    if suite == RESERVED_SUITE:
        return None
    try:
        return _SAFE_NAME.validate_python(suite), _SAFE_NAME.validate_python(run_id)
    except ValidationError:
        return None


def _download_results(*, bucket: storage.Bucket, suite: str, run_id: str, run_dir: Path) -> None:
    """Download a run's results file, leaving none behind when the object is missing."""
    target = run_dir / RESULTS_FILE
    try:
        bucket.blob(f"{suite}/{run_id}/{RESULTS_FILE}").download_to_filename(target)
    except NotFound:
        # suite and run_id are validated safe names, so the log may show them.
        logger.warning("Run %s/%s has no %s object", suite, run_id, RESULTS_FILE)
        target.unlink(missing_ok=True)


def publish(*, bucket: storage.Bucket, root: Path) -> tuple[str, ...]:
    """Download committed runs to an empty working directory and overwrite the HTML page.

    A run whose results object is missing keeps only its manifest, so the page lists it
    as unreadable.

    Returns:
        The runs the page left out as unreadable, as ``<suite>/<run_id>``.

    Raises:
        ValueError: No run is readable; the page is left untouched.
    """
    for manifest in bucket.list_blobs():
        if (run_path := _run_path(manifest.name)) is None:
            continue
        suite, run_id = run_path
        run_dir = root / suite / run_id
        run_dir.mkdir(parents=True)
        manifest.download_to_filename(run_dir / MANIFEST_FILE)
        _download_results(bucket=bucket, suite=suite, run_id=run_id, run_dir=run_dir)

    built = build_page(root=root)
    page = bucket.blob(_PAGE_OBJECT)
    page.cache_control = "private, no-store"
    page.upload_from_string(built.html, content_type="text/html; charset=utf-8")
    return built.unreadable


def main(argv: Sequence[str] | None = None) -> int:
    """Publish to the named bucket using Application Default Credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="Cloud Storage bucket name")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    try:
        bucket = storage.Client().bucket(args.bucket)
        with TemporaryDirectory(prefix="dfo-evals-") as work:
            unreadable = publish(bucket=bucket, root=Path(work))
    except (GoogleAPIError, GoogleAuthError, OSError, ValueError) as error:
        # Validation and transport errors can include record contents.
        logger.error("Could not publish %s (%s)", _PAGE_OBJECT, type(error).__name__)
        return 1

    logger.info("Published gs://%s/%s", args.bucket, _PAGE_OBJECT)
    if unreadable:
        # Run names are safe directory names, so the public log may show them.
        logger.error("The page left out unreadable runs: %s", ", ".join(unreadable))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
