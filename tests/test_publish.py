"""Exercise the publication contract with a stub bucket."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from google.api_core.exceptions import NotFound

from dfo_evals import publish
from dfo_evals.page import Page


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


def test_leaves_the_page_untouched_if_a_committed_records_file_is_missing(tmp_path):
    bucket = Mock()
    bucket.list_blobs.return_value = [_blob("apis-search/nightly-1/manifest.json")]
    bucket.blob.return_value.download_to_filename.side_effect = NotFound("missing results")

    with pytest.raises(NotFound):
        publish.publish(bucket=bucket, root=tmp_path)

    bucket.blob.assert_called_once_with("apis-search/nightly-1/results.jsonl")


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
