"""Thin, retrying wrapper over :class:`sciencebasepy.SbSession`.

This is the only place the rest of the release tooling touches ScienceBase.
It exposes the handful of operations the publish orchestration needs
-- authenticate, look up / create / update an item, find a child by title,
upload a file with a skip-if-unchanged fast path -- and wraps each underlying
call in exponential-backoff retry for transient HTTP failures (429 + 5xx) and
connection/timeout errors.

Design constraints:

- **Import is side-effect-free.** ``sciencebasepy`` is imported lazily inside
  the authenticating factories, never at module load, so ``import sb_client``
  costs nothing and needs no network or credentials. The retry machinery and
  the pure helpers (:func:`remote_file_checksum`, :func:`is_retryable`) are
  usable without ever constructing a session.
- **The sleep is injectable.** Offline tests pass a no-op ``sleep`` so the
  backoff path is exercised without wall-clock delay.
- **No DOI minting.** ScienceBase mints DOIs as a manual staff step; a
  DOI-locked umbrella still accepts new children, so nothing here tries to
  create or refresh a DOI.

sciencebasepy raises bare ``Exception`` objects carrying the HTTP status in
their message (``"Too many requests"`` for 429, ``"Other HTTP error: 503: ..."``
for the rest), so :func:`is_retryable` classifies by message text as well as
by an attached ``response.status_code`` (for ``requests``-level errors).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from nhf_spatial_targets.release._models import ReleaseError

logger = logging.getLogger(__name__)

# HTTP statuses worth retrying: 429 (rate limit) + the transient 5xx family.
# 500/502/503/504 are typically a busy or restarting upstream; 501 (not
# implemented) and other 5xx are not retried.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# sciencebasepy._check_errors wraps non-2xx responses in a bare Exception whose
# text we pattern-match: "Too many requests" (429) and
# "Other HTTP error: <code>: <body>" (the catch-all branch).
_TOO_MANY_REQUESTS_MARKER = "Too many requests"
_OTHER_HTTP_ERROR_RE = re.compile(r"Other HTTP error:\s*(\d{3})")


class SbClientError(ReleaseError):
    """Raised for client-side conditions the wrapper itself detects.

    Distinct from the transport errors raised by sciencebasepy/requests: this
    signals a logical problem the caller must resolve (e.g. an ambiguous title
    match), not a transient failure to retry. A :class:`ReleaseError` subclass
    so the orchestration layer can catch all release-layer errors uniformly.
    """


def _status_from_exception(exc: BaseException) -> int | None:
    """Best-effort HTTP status for *exc*, or ``None`` if not determinable.

    Handles both ``requests``-level errors (status on ``exc.response``) and
    sciencebasepy's message-only exceptions.
    """
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    message = str(exc)
    if _TOO_MANY_REQUESTS_MARKER in message:
        return 429
    match = _OTHER_HTTP_ERROR_RE.search(message)
    if match:
        return int(match.group(1))
    return None


def is_retryable(exc: BaseException) -> bool:
    """Return whether *exc* represents a transient, retryable failure.

    Connection resets and timeouts are retryable; HTTP errors are retryable
    only for :data:`RETRYABLE_STATUS_CODES`. ``SSLError`` and ``ProxyError``
    subclass ``ConnectionError`` but signal *permanent* misconfiguration (bad
    cert, broken proxy), so they are excluded -- retrying them just delays a
    guaranteed failure and buries the real config problem under retry warnings.
    """
    if isinstance(exc, (requests.exceptions.SSLError, requests.exceptions.ProxyError)):
        return False
    if isinstance(
        exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
    ):
        return True
    code = _status_from_exception(exc)
    return code in RETRYABLE_STATUS_CODES


def remote_file_checksum(item: dict | None, filename: str) -> str | None:
    """Return the recorded checksum *value* for *filename* on *item*.

    Searches both the top-level ``files`` list and any facet ``files`` lists,
    returning the first ``checksum.value`` for an entry whose ``name`` matches.
    Returns ``None`` when the item is empty, the file is absent, or no checksum
    is recorded.

    Note: ScienceBase records uploaded-file checksums as **MD5** (see
    ``SbSession._replace_file``), so a SHA-256 passed to
    :meth:`SbClient.upload_file`'s ``skip_if_sha256_matches`` only matches once
    a SHA-256 is recorded on the remote file -- a responsibility of the publish
    orchestration. When the recorded checksum is an MD5 that can't
    match a SHA-256, the file simply re-uploads: correct, just not skipped.
    """
    if not item:
        return None

    def _scan(files: list[dict] | None) -> str | None:
        for entry in files or []:
            if entry.get("name") == filename:
                checksum = entry.get("checksum") or {}
                value = checksum.get("value")
                if value is not None:
                    return value
        return None

    value = _scan(item.get("files"))
    if value is not None:
        return value
    for facet in item.get("facets") or []:
        value = _scan(facet.get("files"))
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class UploadResult:
    """Outcome of :meth:`SbClient.upload_file`.

    ``skipped`` is ``True`` when the remote checksum already matched and no
    bytes were sent; ``item`` is the ScienceBase item JSON (post-upload when a
    file was sent, pre-upload when skipped).
    """

    skipped: bool
    name: str
    item: dict


def _new_session(env: str | None):
    """Construct a fresh :class:`sciencebasepy.SbSession` (lazy import)."""
    import sciencebasepy

    return sciencebasepy.SbSession(env)


class SbClient:
    """Retrying facade over a :class:`sciencebasepy.SbSession`.

    Construct via :meth:`login` or :meth:`from_token` for a real session, or
    pass an already-built (or mock) session directly to the constructor.

    Parameters
    ----------
    session
        An object exposing the :class:`sciencebasepy.SbSession` surface used
        here (``ping``, ``get_item``, ``create_item``, ``update_item``,
        ``get_child_ids``, ``upload_file_to_item``, ``is_logged_in``). Tests
        inject a mock.
    max_retries
        Maximum number of *retries* after the initial attempt for a retryable
        error. ``0`` disables retrying.
    base_delay, max_delay
        Exponential backoff is ``min(max_delay, base_delay * 2 ** attempt)``
        seconds, where ``attempt`` counts from ``0`` for the first retry.
    sleep
        Sleep callable, injectable so offline tests don't block. Defaults to
        :func:`time.sleep`.
    """

    def __init__(
        self,
        session: Any,
        *,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0; got {max_retries}.")
        if base_delay <= 0 or max_delay <= 0:
            raise ValueError(
                f"base_delay and max_delay must be > 0; got "
                f"base_delay={base_delay}, max_delay={max_delay}."
            )
        self._session = session
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep

    # -- construction ------------------------------------------------------

    @classmethod
    def login(
        cls,
        username: str,
        password: str,
        *,
        env: str | None = None,
        **kwargs: Any,
    ) -> SbClient:
        """Authenticate with a username/password and return a client."""
        session = _new_session(env)
        session.login(username, password)
        return cls(session, **kwargs)

    @classmethod
    def from_token(
        cls,
        token: str,
        *,
        env: str | None = None,
        **kwargs: Any,
    ) -> SbClient:
        """Authenticate with a Keycloak token JSON and return a client."""
        session = _new_session(env)
        session.add_token(token)
        return cls(session, **kwargs)

    @property
    def session(self) -> Any:
        """The wrapped session (escape hatch for advanced operations)."""
        return self._session

    # -- retry core --------------------------------------------------------

    def _backoff_delay(self, attempt: int) -> float:
        return min(self._max_delay, self._base_delay * (2**attempt))

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke *fn* with exponential-backoff retry on transient failures.

        Re-raises the last exception once retries are exhausted or the error
        is non-retryable, preserving the original traceback for the operator.
        """
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if attempt >= self._max_retries or not is_retryable(exc):
                    raise
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "sb_client: retryable error on %s (retry %d/%d in %.1fs): %s",
                    getattr(fn, "__name__", repr(fn)),
                    attempt + 1,
                    self._max_retries,
                    delay,
                    exc,
                )
                self._sleep(delay)
                attempt += 1

    # -- auth / liveness ---------------------------------------------------

    def ping(self) -> Any:
        """Low-cost ScienceBase liveness check (retried)."""
        return self._call(self._session.ping)

    def whoami(self) -> dict:
        """Return ``{"username", "logged_in"}`` from local session state.

        No network call: ``logged_in`` reflects the local token's validity.
        """
        return {
            "username": getattr(self._session, "_username", None),
            "logged_in": bool(self._session.is_logged_in()),
        }

    # -- items -------------------------------------------------------------

    def get_item(self, sb_id: str) -> dict:
        """Fetch the ScienceBase item JSON for *sb_id* (retried)."""
        return self._call(self._session.get_item, sb_id)

    def create_item(self, parent_id: str, body: dict) -> dict:
        """Create a child item under *parent_id* and return its JSON.

        *body* is copied and stamped with ``parentId``; an ``id`` in *body* is
        ignored by ScienceBase on create.
        """
        item = dict(body)
        item["parentId"] = parent_id
        return self._call(self._session.create_item, item)

    def update_item(self, sb_id: str, body: dict) -> dict:
        """Update item *sb_id* with *body* and return the updated JSON.

        *body* is copied and stamped with ``id`` so the caller never has to
        thread the id through the body dict.
        """
        item = dict(body)
        item["id"] = sb_id
        return self._call(self._session.update_item, item)

    def find_child(self, parent_id: str, title: str) -> dict | None:
        """Return the child of *parent_id* whose title equals *title* exactly.

        Returns ``None`` when no child matches. Raises :class:`SbClientError`
        when more than one child shares the exact title -- the publish layer
        can't safely pick one, so the operator must rename or merge the
        duplicates. (Title comparison is exact, not the fuzzy Lucene match
        ScienceBase search would do.)
        """
        child_ids = self._call(self._session.get_child_ids, parent_id)
        matches = []
        for child_id in child_ids:
            item = self._call(self._session.get_item, child_id)
            if item.get("title") == title:
                matches.append(item)
        if len(matches) > 1:
            raise SbClientError(
                f"{len(matches)} children of {parent_id} share the exact "
                f"title {title!r}; cannot disambiguate. Rename or merge the "
                f"duplicates before publishing."
            )
        return matches[0] if matches else None

    def upload_file(
        self,
        sb_id: str,
        path: str | Path,
        *,
        skip_if_sha256_matches: str | None = None,
        scrape_file: bool = True,
    ) -> UploadResult:
        """Upload *path* to item *sb_id*, replacing any same-named file.

        When *skip_if_sha256_matches* is given and the remote item already
        carries a file with the same basename whose recorded checksum value
        equals it, the upload is skipped (no bytes sent) and an
        :class:`UploadResult` with ``skipped=True`` is returned. This is the
        partial-upload / re-publish fast path: a re-run skips files that are
        already present and current.

        Note: ScienceBase records MD5 checksums; see
        :func:`remote_file_checksum` for why a SHA-256 only matches once the
        orchestration layer records one.
        """
        path = Path(path)
        item = self._call(self._session.get_item, sb_id)
        if skip_if_sha256_matches is not None:
            remote = remote_file_checksum(item, path.name)
            if remote is not None and remote == skip_if_sha256_matches:
                logger.info(
                    "sb_client: skipping upload of %s to %s (checksum match)",
                    path.name,
                    sb_id,
                )
                return UploadResult(skipped=True, name=path.name, item=item)
        updated = self._call(
            self._session.upload_file_to_item, item, str(path), scrape_file
        )
        return UploadResult(skipped=False, name=path.name, item=updated)

    def delete_file(self, sb_id: str, name: str) -> dict:
        """Delete every file named *name* from item *sb_id*; return its JSON.

        Fetches the item, then dispatches to
        :meth:`sciencebasepy.SbSession.delete_file`, which drops all matching
        entries from both the top-level ``files`` list and any facet ``files``
        lists and PUTs the trimmed item back. Used by the publish layer's
        ``--delete-orphans`` path to remove files left on a ScienceBase item
        that the current staged payload no longer contains.
        """
        item = self._call(self._session.get_item, sb_id)
        return self._call(self._session.delete_file, name, item)
