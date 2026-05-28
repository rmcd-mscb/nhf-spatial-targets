"""Offline tests for ``nhf_spatial_targets.release.sb_client``.

No real network: every test injects a mock session and a no-op sleep, so the
retry/backoff path is exercised without wall-clock delay. Live ScienceBase
tests are gated behind ``@pytest.mark.integration`` and live in a later PR.

The contracts locked here:

- :func:`is_retryable` classifies 429 + transient 5xx + connection/timeout as
  retryable and everything else as fatal -- matching the bare-``Exception``
  messages sciencebasepy actually raises,
- ``_call`` retries with exponential backoff using the injected sleep and
  re-raises once retries are exhausted,
- ``upload_file`` skips the byte transfer when the remote checksum matches,
- ``create_item`` / ``update_item`` / ``find_child`` dispatch to the right
  underlying session calls with the id/parentId stamped correctly.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from nhf_spatial_targets.release import sb_client
from nhf_spatial_targets.release.sb_client import (
    SbClient,
    SbClientError,
    UploadResult,
    is_retryable,
    remote_file_checksum,
)


@pytest.fixture
def sleeps() -> list[float]:
    """Collect the durations passed to the injected sleep."""
    return []


@pytest.fixture
def client_factory(sleeps: list[float]):
    """Build an SbClient over a MagicMock session with a recording sleep."""

    def _make(session=None, **kwargs):
        session = session if session is not None else MagicMock()
        kwargs.setdefault("base_delay", 0.5)
        kwargs.setdefault("max_delay", 100.0)
        kwargs.setdefault("sleep", sleeps.append)
        return SbClient(session, **kwargs)

    return _make


# ---------------------------------------------------------------------------
# is_retryable classification
# ---------------------------------------------------------------------------


def test_too_many_requests_is_retryable() -> None:
    """sciencebasepy raises ``Exception('Too many requests')`` for 429."""
    assert is_retryable(Exception("Too many requests")) is True


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_transient_5xx_is_retryable(code: int) -> None:
    """sciencebasepy's catch-all ``Other HTTP error: <code>`` for 5xx."""
    assert is_retryable(Exception(f"Other HTTP error: {code}: upstream busy"))


@pytest.mark.parametrize("code", [400, 401, 403, 404, 501])
def test_non_retryable_status_codes(code: int) -> None:
    """Client errors and 501 are fatal, not retried."""
    assert is_retryable(Exception(f"Other HTTP error: {code}: nope")) is False


def test_unauthorized_message_is_not_retryable() -> None:
    assert is_retryable(Exception("Unauthorized access")) is False


def test_connection_and_timeout_errors_are_retryable() -> None:
    assert is_retryable(requests.exceptions.ConnectionError("reset")) is True
    assert is_retryable(requests.exceptions.Timeout("slow")) is True


def test_permanent_connection_subclasses_are_not_retryable() -> None:
    """SSLError/ProxyError subclass ConnectionError but are permanent config
    failures -- retrying them just buries the real problem.
    """
    assert is_retryable(requests.exceptions.SSLError("bad cert")) is False
    assert is_retryable(requests.exceptions.ProxyError("bad proxy")) is False


def test_http_error_with_response_status_is_retryable() -> None:
    """A requests.HTTPError carrying a 503 response is retryable."""
    exc = requests.exceptions.HTTPError("boom")
    exc.response = SimpleNamespace(status_code=503)
    assert is_retryable(exc) is True
    exc.response = SimpleNamespace(status_code=404)
    assert is_retryable(exc) is False


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------


def test_retry_on_429_then_success(client_factory, sleeps: list[float]) -> None:
    """Two 429s then a success: 3 calls, 2 backoff sleeps, value returned."""
    session = MagicMock()
    session.ping.side_effect = [
        Exception("Too many requests"),
        Exception("Too many requests"),
        {"ok": True},
    ]
    client = client_factory(session, max_retries=5)

    result = client.ping()

    assert result == {"ok": True}
    assert session.ping.call_count == 3
    # Exponential: base_delay=0.5 -> [0.5, 1.0].
    assert sleeps == [0.5, 1.0]


def test_retry_on_503_then_success(client_factory, sleeps: list[float]) -> None:
    session = MagicMock()
    session.get_item.side_effect = [
        Exception("Other HTTP error: 503: service unavailable"),
        {"id": "abc", "title": "T"},
    ]
    client = client_factory(session, max_retries=3)

    result = client.get_item("abc")

    assert result["id"] == "abc"
    assert session.get_item.call_count == 2
    assert sleeps == [0.5]


def test_non_retryable_raises_immediately(client_factory, sleeps: list[float]) -> None:
    """A fatal error is raised on the first attempt with no sleep."""
    session = MagicMock()
    session.get_item.side_effect = Exception("Unauthorized access")
    client = client_factory(session, max_retries=5)

    with pytest.raises(Exception, match="Unauthorized access"):
        client.get_item("abc")
    assert session.get_item.call_count == 1
    assert sleeps == []


def test_retries_exhausted_reraises(client_factory, sleeps: list[float]) -> None:
    """After max_retries the last exception propagates."""
    session = MagicMock()
    session.ping.side_effect = Exception("Too many requests")
    client = client_factory(session, max_retries=2)

    with pytest.raises(Exception, match="Too many requests"):
        client.ping()
    # initial attempt + 2 retries = 3 calls; 2 sleeps.
    assert session.ping.call_count == 3
    assert sleeps == [0.5, 1.0]


def test_max_delay_caps_backoff(client_factory, sleeps: list[float]) -> None:
    """Backoff never exceeds max_delay."""
    session = MagicMock()
    session.ping.side_effect = Exception("Too many requests")
    client = client_factory(session, max_retries=4, base_delay=1.0, max_delay=3.0)

    with pytest.raises(Exception, match="Too many requests"):
        client.ping()
    # 1, 2, 4->capped 3, 8->capped 3
    assert sleeps == [1.0, 2.0, 3.0, 3.0]


def test_max_retries_zero_disables_retry(client_factory, sleeps: list[float]) -> None:
    session = MagicMock()
    session.ping.side_effect = Exception("Too many requests")
    client = client_factory(session, max_retries=0)

    with pytest.raises(Exception, match="Too many requests"):
        client.ping()
    assert session.ping.call_count == 1
    assert sleeps == []


# ---------------------------------------------------------------------------
# Item dispatch: get / create / update
# ---------------------------------------------------------------------------


def test_get_item_dispatches(client_factory) -> None:
    session = MagicMock()
    session.get_item.return_value = {"id": "abc", "title": "T"}
    client = client_factory(session)

    assert client.get_item("abc") == {"id": "abc", "title": "T"}
    session.get_item.assert_called_once_with("abc")


def test_create_item_stamps_parent_id(client_factory) -> None:
    """create_item adds parentId and does not mutate the caller's body."""
    session = MagicMock()
    session.create_item.return_value = {"id": "new", "title": "T"}
    client = client_factory(session)

    body = {"title": "T", "summary": "s"}
    result = client.create_item("parent123", body)

    assert result["id"] == "new"
    session.create_item.assert_called_once_with(
        {"title": "T", "summary": "s", "parentId": "parent123"}
    )
    assert "parentId" not in body  # caller's dict untouched


def test_update_item_stamps_id(client_factory) -> None:
    session = MagicMock()
    session.update_item.return_value = {"id": "abc", "title": "T2"}
    client = client_factory(session)

    body = {"title": "T2"}
    result = client.update_item("abc", body)

    assert result["title"] == "T2"
    session.update_item.assert_called_once_with({"title": "T2", "id": "abc"})
    assert "id" not in body


# ---------------------------------------------------------------------------
# find_child
# ---------------------------------------------------------------------------


def test_find_child_returns_exact_title_match(client_factory) -> None:
    session = MagicMock()
    session.get_child_ids.return_value = ["c1", "c2"]
    session.get_item.side_effect = lambda cid: {
        "c1": {"id": "c1", "title": "ERA5-Land"},
        "c2": {"id": "c2", "title": "MERRA-2"},
    }[cid]
    client = client_factory(session)

    match = client.find_child("parent", "MERRA-2")
    assert match["id"] == "c2"
    session.get_child_ids.assert_called_once_with("parent")


def test_find_child_returns_none_when_no_match(client_factory) -> None:
    session = MagicMock()
    session.get_child_ids.return_value = ["c1"]
    session.get_item.return_value = {"id": "c1", "title": "Other"}
    client = client_factory(session)

    assert client.find_child("parent", "ERA5-Land") is None


def test_find_child_raises_on_ambiguous_titles(client_factory) -> None:
    """Two children with the same exact title is unrecoverable here."""
    session = MagicMock()
    session.get_child_ids.return_value = ["c1", "c2"]
    session.get_item.side_effect = lambda cid: {"id": cid, "title": "DUP"}
    client = client_factory(session)

    with pytest.raises(SbClientError, match="disambiguate"):
        client.find_child("parent", "DUP")


# ---------------------------------------------------------------------------
# upload_file: skip-if-checksum-matches
# ---------------------------------------------------------------------------


def _item_with_file(name: str, sha: str) -> dict:
    return {
        "id": "child1",
        "files": [{"name": name, "checksum": {"type": "SHA-256", "value": sha}}],
    }


def test_upload_file_skips_when_checksum_matches(client_factory, tmp_path) -> None:
    """A remote file with a matching checksum means no bytes are sent."""
    f = tmp_path / "data.nc"
    f.write_bytes(b"payload")
    sha = "abc123sha"

    session = MagicMock()
    session.get_item.return_value = _item_with_file("data.nc", sha)
    client = client_factory(session)

    result = client.upload_file("child1", f, skip_if_sha256_matches=sha)

    assert isinstance(result, UploadResult)
    assert result.skipped is True
    assert result.name == "data.nc"
    session.upload_file_to_item.assert_not_called()


def test_upload_file_uploads_when_checksum_differs(client_factory, tmp_path) -> None:
    f = tmp_path / "data.nc"
    f.write_bytes(b"payload")

    session = MagicMock()
    session.get_item.return_value = _item_with_file("data.nc", "OLDsha")
    session.upload_file_to_item.return_value = {"id": "child1", "files": []}
    client = client_factory(session)

    result = client.upload_file("child1", f, skip_if_sha256_matches="NEWsha")

    assert result.skipped is False
    session.upload_file_to_item.assert_called_once()
    # the staged file path is forwarded
    args = session.upload_file_to_item.call_args.args
    assert args[1] == str(f)


def test_upload_file_uploads_when_no_skip_requested(client_factory, tmp_path) -> None:
    f = tmp_path / "data.nc"
    f.write_bytes(b"payload")

    session = MagicMock()
    session.get_item.return_value = {"id": "child1", "files": []}
    session.upload_file_to_item.return_value = {"id": "child1"}
    client = client_factory(session)

    result = client.upload_file("child1", f)

    assert result.skipped is False
    session.upload_file_to_item.assert_called_once()


def test_upload_file_uploads_when_remote_file_absent(client_factory, tmp_path) -> None:
    """skip requested but the remote item has no such file: upload anyway."""
    f = tmp_path / "data.nc"
    f.write_bytes(b"payload")

    session = MagicMock()
    session.get_item.return_value = {"id": "child1", "files": []}
    session.upload_file_to_item.return_value = {"id": "child1"}
    client = client_factory(session)

    result = client.upload_file("child1", f, skip_if_sha256_matches="whatever")
    assert result.skipped is False
    session.upload_file_to_item.assert_called_once()


# ---------------------------------------------------------------------------
# remote_file_checksum helper
# ---------------------------------------------------------------------------


def test_remote_file_checksum_in_files() -> None:
    item = _item_with_file("a.nc", "SHA-A")
    assert remote_file_checksum(item, "a.nc") == "SHA-A"


def test_remote_file_checksum_in_facets() -> None:
    item = {
        "files": [],
        "facets": [
            {"files": [{"name": "b.nc", "checksum": {"value": "SHA-B"}}]},
        ],
    }
    assert remote_file_checksum(item, "b.nc") == "SHA-B"


def test_remote_file_checksum_absent_returns_none() -> None:
    assert remote_file_checksum({"files": []}, "missing.nc") is None
    assert remote_file_checksum(None, "missing.nc") is None
    # present but no checksum recorded
    assert remote_file_checksum({"files": [{"name": "c.nc"}]}, "c.nc") is None


# ---------------------------------------------------------------------------
# whoami + factories (offline)
# ---------------------------------------------------------------------------


def test_whoami_reads_local_session_state(client_factory) -> None:
    session = MagicMock()
    session._username = "jdoe"
    session.is_logged_in.return_value = True
    client = client_factory(session)

    assert client.whoami() == {"username": "jdoe", "logged_in": True}


def test_whoami_tolerates_missing_username() -> None:
    """A token session may not set ``_username``; whoami yields None, not raise.

    (Uses a bare object rather than MagicMock, whose attribute auto-vivifies.)
    """

    class _Session:
        def is_logged_in(self):
            return False

    client = SbClient(_Session(), sleep=lambda _: None)
    assert client.whoami() == {"username": None, "logged_in": False}


def test_login_factory_dispatches(monkeypatch) -> None:
    """login() builds a session and calls session.login (no network)."""
    fake = MagicMock()
    monkeypatch.setattr(sb_client, "_new_session", lambda env: fake)

    client = SbClient.login("user", "pass", env="beta", max_retries=2)

    assert isinstance(client, SbClient)
    assert client.session is fake
    fake.login.assert_called_once_with("user", "pass")


def test_from_token_factory_dispatches(monkeypatch) -> None:
    fake = MagicMock()
    monkeypatch.setattr(sb_client, "_new_session", lambda env: fake)

    client = SbClient.from_token("token-json", env="beta")

    assert client.session is fake
    fake.add_token.assert_called_once_with("token-json")
