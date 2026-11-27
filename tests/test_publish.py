"""Exercise the publication contract with a stub bucket."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
from google.api_core.exceptions import Forbidden, NotFound

from dfo_evals import publish
from dfo_evals.page import Page
from dfo_evals.records import Run, RunManifest, write_run


def _blob(name: str, content: str = "") -> Mock:
    blob = Mock()
    blob.name = name
    blob.download_to_filename.side_effect = lambda path: Path(path).write_text(
        content, encoding="utf-8"
    )
    return blob


def test_downloads_complete_runs_and_publishes_only_the_page(tmp_path, monkeypatch):
    manifest = _blob("apis-search/nightly-1/manifest.json", "manifest")
    incomplete = _blob("apis-search/nightly-2/results.jsonl")
    invalid = _blob("apis-search/../manifest.json")
    reserved = _blob("site/nightly-1/manifest.json")
    results = _blob("apis-search/nightly-1/results.jsonl", "results")
    page = _blob("site/index.html")
    bucket = Mock()
    bucket.list_blobs.return_value = [manifest, incomplete, invalid, reserved, page]
    bucket.blob.side_effect = {results.name: results, page.name: page}.__getitem__

    def build_page(*, root):
        run = root / "apis-search" / "nightly-1"
        assert sorted(root.rglob("*.json*")) == [run / "manifest.json", run / "results.jsonl"]
        assert (run / "manifest.json").read_text(encoding="utf-8") == "manifest"
        assert (run / "results.jsonl").read_text(encoding="utf-8") == "results"
        return Page(html="<html>report</html>", unreadable=())

    monkeypatch.setattr(publish, "build_page", build_page)
    assert publish.publish(bucket=bucket, root=tmp_path) == ()

    assert page.cache_control == "private, no-store"
    page.upload_from_string.assert_called_once_with(
        "<html>report</html>", content_type="text/html; charset=utf-8"
    )


def _stored_run(source: Path, run_id: str) -> dict[str, str]:
    """Write a valid run under ``source`` and return its objects by bucket name."""
    manifest = RunManifest(
        suite="apis-search",
        run_id=run_id,
        started_at=datetime(2026, 11, 1, tzinfo=UTC),
        finished_at=None,
        revision="rev1",
        environment={},
        dataset_digest="sha256:dataset",
    )
    run_dir = write_run(root=source, run=Run(manifest=manifest, results=()))
    return {
        f"apis-search/{run_id}/{path.name}": path.read_text(encoding="utf-8")
        for path in run_dir.iterdir()
    }


def _bucket(objects: dict[str, str], *, missing: str) -> tuple[Mock, Mock]:
    """A stub bucket serving ``objects`` whose ``missing`` object raises NotFound."""
    blobs = {name: _blob(name, content) for name, content in objects.items()}
    blobs[missing] = _blob(missing)
    blobs[missing].download_to_filename.side_effect = NotFound("missing results")
    page = _blob("site/index.html")
    blobs[page.name] = page
    bucket = Mock()
    bucket.list_blobs.return_value = [blob for blob in blobs.values() if blob.name != missing]
    bucket.blob.side_effect = blobs.__getitem__
    return bucket, page


def test_a_run_missing_its_records_file_is_listed_and_the_page_published(tmp_path, caplog):
    objects = _stored_run(tmp_path / "source", "nightly-1")
    objects["apis-search/nightly-2/manifest.json"] = _stored_run(tmp_path / "source", "nightly-2")[
        "apis-search/nightly-2/manifest.json"
    ]
    bucket, page = _bucket(objects, missing="apis-search/nightly-2/results.jsonl")
    root = tmp_path / "work"

    assert publish.publish(bucket=bucket, root=root) == ("apis-search/nightly-2",)

    assert sorted(path.name for path in (root / "apis-search" / "nightly-2").iterdir()) == [
        "manifest.json"
    ]
    assert "apis-search/nightly-2" in caplog.text
    page.upload_from_string.assert_called_once()
    assert "<li><code>apis-search/nightly-2</code></li>" in page.upload_from_string.call_args[0][0]


def test_leaves_the_page_untouched_if_no_run_has_its_records_file(tmp_path):
    objects = _stored_run(tmp_path / "source", "nightly-1")
    bucket, page = _bucket(
        {name: text for name, text in objects.items() if name.endswith("manifest.json")},
        missing="apis-search/nightly-1/results.jsonl",
    )

    with pytest.raises(ValueError, match="apis-search/nightly-1"):
        publish.publish(bucket=bucket, root=tmp_path / "work")

    page.upload_from_string.assert_not_called()


def test_main_fails_when_a_run_is_missing_its_records_file(tmp_path, monkeypatch):
    objects = _stored_run(tmp_path / "source", "nightly-1")
    objects["apis-search/nightly-2/manifest.json"] = _stored_run(tmp_path / "source", "nightly-2")[
        "apis-search/nightly-2/manifest.json"
    ]
    bucket, page = _bucket(objects, missing="apis-search/nightly-2/results.jsonl")
    client = Mock()
    client.return_value.bucket.return_value = bucket
    monkeypatch.setattr(publish.storage, "Client", client)

    assert publish.main(["--bucket", "bucket"]) == 1
    page.upload_from_string.assert_called_once()


def test_leaves_the_page_untouched_if_listing_the_bucket_fails(tmp_path):
    bucket = Mock()
    bucket.list_blobs.side_effect = Forbidden("no list permission")

    with pytest.raises(Forbidden):
        publish.publish(bucket=bucket, root=tmp_path)

    bucket.blob.assert_not_called()


@pytest.mark.parametrize(("unreadable", "code"), [((), 0), (("apis-search/nightly-2",), 1)])
def test_main_publishes_and_fails_when_the_page_left_runs_out(
    monkeypatch, caplog, unreadable, code
):
    monkeypatch.setattr(publish.storage, "Client", Mock())
    monkeypatch.setattr(publish, "publish", lambda *, bucket, root: unreadable)

    assert publish.main(["--bucket", "bucket"]) == code
    assert ("apis-search/nightly-2" in caplog.text) == bool(unreadable)


def test_publish_returns_the_runs_the_page_left_out(tmp_path, monkeypatch):
    bucket = Mock()
    bucket.list_blobs.return_value = []
    left_out = ("apis-search/nightly-2",)
    monkeypatch.setattr(
        publish, "build_page", lambda *, root: Page(html="<html></html>", unreadable=left_out)
    )

    assert publish.publish(bucket=bucket, root=tmp_path) == left_out
    bucket.blob.return_value.upload_from_string.assert_called_once()
