"""
Small sync and async HTTP client for the Jobo API.

Internal. It owns the key, the headers and the retry and error rules:

- headers: `X-Api-Key`, a `User-Agent` and `X-Jobo-Client: llamaindex`;
- connections: one `httpx.Client` and one `httpx.AsyncClient` per event loop
  are kept and reused, and closed by `close()` / `aclose()`;
- retries: only 429, 500 and 503 and failures to connect are retried,
  `Retry-After` is honoured, and 402 is never retried. A read timeout is not
  retried either, because a billed page may already have been delivered. A
  503 without `Retry-After` is not retried, and neither is a 500 on a billed
  read (search and hiring signals);
- errors: every non-2xx response is raised as a typed `JoboAPIError`.
"""

import asyncio
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import httpx

from llama_index.tools.jobo.errors import (
    JoboAPIError,
    JoboAuthenticationError,
    JoboConnectionError,
    JoboError,
    JoboInsufficientCreditsError,
    JoboNotFoundError,
    JoboPermissionError,
    JoboRateLimitError,
    JoboServerError,
    JoboValidationError,
    field_errors,
)

__version__ = "0.3.0"

DEFAULT_BASE_URL = "https://connect.jobo.world"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 2
#: A `Retry-After` longer than this is not slept through: the error is raised.
MAX_RETRY_WAIT = 60.0
API_KEY_ENV = "JOBO_API_KEY"
API_KEYS_URL = "https://enterprise.jobo.world/api-keys"
CLIENT_NAME = "llamaindex"
USER_AGENT = f"llama-index-tools-jobo/{__version__}"

_RETRYABLE_STATUSES = frozenset({429, 500, 503})
#: `POST /api/jobs/search` bills every job it returns, and the hiring signals
#: reads every signal, every time. Both take the charge before they build the
#: response: a 500 can follow a charge, and a replay would charge again. Only
#: a refusal that comes before any work is retried there.
_BILLED_RETRYABLE_STATUSES = frozenset({429, 503})
_BILLED_SEARCH_PATH = "/api/jobs/search"
#: `GET /api/signals` and `GET /api/companies/{id}/signals`.
_BILLED_SIGNALS_PATH = re.compile(r"/api/signals|/api/companies/[^/]+/signals")

_ERROR_BY_STATUS: Dict[int, type] = {
    400: JoboValidationError,
    401: JoboAuthenticationError,
    402: JoboInsufficientCreditsError,
    403: JoboPermissionError,
    404: JoboNotFoundError,
    429: JoboRateLimitError,
}


def resolve_api_key(api_key: Optional[str] = None) -> str:
    """Return the key to use, trimmed, or raise when there is none."""
    value = (
        api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
    ).strip()
    if not value:
        raise ValueError(
            f"A Jobo API key is required. Pass api_key=... or set the {API_KEY_ENV} "
            f"environment variable. Create a key at {API_KEYS_URL}"
        )
    return value


@dataclass(frozen=True)
class Reply:
    """A successful response: the parsed JSON body and the metering headers."""

    data: Any
    usage: Dict[str, Any] = field(default_factory=dict)


_USAGE_HEADERS = {
    "credits_deducted": "x-credits-deducted",
    "credits_balance": "x-credits-balance",
    "quota_limit": "x-quota-limit",
    "quota_remaining": "x-quota-remaining",
}


def _usage(headers: Mapping[str, str]) -> Dict[str, Any]:
    """The metering the API reported. Only values it actually sent."""
    usage: Dict[str, Any] = {}
    for name, header in _USAGE_HEADERS.items():
        raw = headers.get(header)
        if raw is not None:
            try:
                usage[name] = int(raw.strip())
            except ValueError:
                continue
    status = headers.get("x-billing-status")
    if status:
        usage["billing_status"] = status
    return usage


class _Attempt(Exception):
    """An attempt that failed, and whether another attempt may help."""

    def __init__(self, error: JoboError, retryable: bool) -> None:
        super().__init__(str(error))
        self.error = error
        self.retryable = retryable


class JoboClient:
    """Minimal Jobo API client. The key is only ever written into a header."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: Optional[httpx.Client] = None,
        http_async_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be 0 or greater.")
        self._api_key = resolve_api_key(api_key)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._http_client = http_client
        self._http_async_client = http_async_client
        self._owned_client: Optional[httpx.Client] = None
        self._owned_client_lock = threading.Lock()
        # An `httpx.AsyncClient` belongs to the event loop it first ran on, so
        # one is kept per loop, until that loop has closed.
        self._owned_async_clients: Dict[
            asyncio.AbstractEventLoop, httpx.AsyncClient
        ] = {}
        self._sleep = time.sleep
        self._asleep = asyncio.sleep

    def __repr__(self) -> str:
        return f"JoboClient(base_url={self.base_url!r})"

    def is_sandbox_key(self) -> bool:
        """Whether the key is a sandbox key (`jbe_test_...`)."""
        return self._api_key.startswith("jbe_test_")

    def _kwargs(
        self,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]],
        json: Any,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "method": method,
            "url": f"{self.base_url}{path}",
            "headers": {
                "X-Api-Key": self._api_key,
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
                "X-Jobo-Client": CLIENT_NAME,
            },
            "timeout": self.timeout,
        }
        if params:
            kwargs["params"] = dict(params)
        if json is not None:
            kwargs["json"] = json
        return kwargs

    def _delay(self, failure: _Attempt, attempt: int) -> Optional[float]:
        """Seconds to wait before the next attempt, or None to give up."""
        if not failure.retryable or attempt >= self.max_retries:
            return None
        error = failure.error
        retry_after = error.retry_after if isinstance(error, JoboAPIError) else None
        if retry_after is not None:
            delay = max(retry_after, 0.0)
        else:
            delay = min(float(2**attempt), 30.0) * random.uniform(0.75, 1.25)
        return delay if delay <= MAX_RETRY_WAIT else None

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Any = None,
    ) -> Reply:
        """Send a request and return the parsed JSON body and metering."""
        kwargs = self._kwargs(method, path, params, json)
        billed = _is_billed(method, path)
        attempt = 0
        while True:
            try:
                return self._attempt(kwargs, billed)
            except _Attempt as failure:
                delay = self._delay(failure, attempt)
                if delay is None:
                    raise failure.error from failure.__cause__
                self._sleep(delay)
                attempt += 1

    def _attempt(self, kwargs: Dict[str, Any], billed: bool) -> Reply:
        try:
            response = self._sync_client().request(**kwargs)
        except httpx.TransportError as exc:
            raise _transport_failure(exc) from exc
        return _handle(response, billed)

    def _sync_client(self) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        with self._owned_client_lock:
            if self._owned_client is None:
                self._owned_client = httpx.Client()
            return self._owned_client

    def close(self) -> None:
        """
        Close the sync HTTP client this instance created, if it created one.

        A client passed in as `http_client` is left open. The instance stays
        usable: the next request opens a new client.
        """
        with self._owned_client_lock:
            client, self._owned_client = self._owned_client, None
        if client is not None:
            client.close()

    async def arequest(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Any = None,
    ) -> Reply:
        """Async twin of `request`."""
        kwargs = self._kwargs(method, path, params, json)
        billed = _is_billed(method, path)
        client = self._async_client()
        attempt = 0
        while True:
            try:
                try:
                    response = await client.request(**kwargs)
                except httpx.TransportError as exc:
                    raise _transport_failure(exc) from exc
                return _handle(response, billed)
            except _Attempt as failure:
                delay = self._delay(failure, attempt)
                if delay is None:
                    raise failure.error from failure.__cause__
                await self._asleep(delay)
                attempt += 1

    def _async_client(self) -> httpx.AsyncClient:
        """The client for the running event loop, opened on first use."""
        if self._http_async_client is not None:
            return self._http_async_client
        loop = asyncio.get_running_loop()
        client = self._owned_async_clients.get(loop)
        if client is None:
            # A client whose loop has closed can neither be used nor awaited
            # shut: let go of it, so repeated `asyncio.run()` calls do not
            # pile up one dead client each.
            for closed in [o for o in self._owned_async_clients if o.is_closed()]:
                del self._owned_async_clients[closed]
            client = httpx.AsyncClient()
            self._owned_async_clients[loop] = client
        return client

    async def aclose(self) -> None:
        """
        Close the HTTP clients this instance created.

        Closes the async client of the running event loop and the sync client.
        Clients passed in as `http_client` / `http_async_client` are left
        open. The instance stays usable: the next request opens a new client.
        """
        client = self._owned_async_clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            await client.aclose()
        self.close()


def _transport_failure(exc: httpx.TransportError) -> _Attempt:
    # Only a failure to connect is certain not to have reached the API.
    retryable = isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))
    error = JoboConnectionError(
        f"Could not get a response from the Jobo API: {type(exc).__name__}: {exc}"
    )
    return _Attempt(error, retryable)


def _is_billed(method: str, path: str) -> bool:
    """Whether the request bills what it returns: a search, or hiring signals."""
    verb = method.upper()
    if verb == "POST":
        return path == _BILLED_SEARCH_PATH
    return verb == "GET" and _BILLED_SIGNALS_PATH.fullmatch(path) is not None


def _is_retryable(error: JoboAPIError, billed: bool) -> bool:
    status = error.status_code
    if status not in (_BILLED_RETRYABLE_STATUSES if billed else _RETRYABLE_STATUSES):
        return False
    # The API sends `Retry-After` with a 503 it expects to pass ("unavailable").
    # A 503 without it, such as "Search took too long", fails the same way
    # again, and each replay holds a search thread on the server.
    return status != 503 or error.retry_after is not None


def _handle(response: httpx.Response, billed: bool) -> Reply:
    if response.is_success:
        try:
            return Reply(response.json(), _usage(response.headers))
        except ValueError as exc:
            error = JoboError(
                f"The Jobo API answered HTTP {response.status_code} with a body "
                "that is not valid JSON."
            )
            raise _Attempt(error, False) from exc
    api_error = _error_from_response(response)
    raise _Attempt(api_error, _is_retryable(api_error, billed))


def _to_seconds(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _error_from_response(response: httpx.Response) -> JoboAPIError:
    status = response.status_code
    body: Any
    try:
        body = response.json()
    except ValueError:
        body = response.text

    code: Optional[str] = None
    detail: Optional[str] = None
    retry_after = _to_seconds(response.headers.get("retry-after"))
    if isinstance(body, dict):
        raw_code = body.get("code")
        code = raw_code if isinstance(raw_code, str) and raw_code else None
        for key in ("detail", "error", "title", "message"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                detail = value.strip()
                break
        if retry_after is None:
            retry_after = _to_seconds(body.get("retry_after_seconds"))

    message = f"Jobo API error {status}"
    if code:
        message += f" ({code})"
    if detail:
        message += f": {detail}"
    fields = "; ".join(
        f"{field}: {'; '.join(text.rstrip('.') for text in texts)}"
        for field, texts in field_errors(body).items()
    )
    if fields:
        message += f". {fields}."
    if status == 429 and retry_after is not None:
        message += f" Retry after {retry_after:g} seconds."

    error_cls = _ERROR_BY_STATUS.get(status)
    if error_cls is None:
        error_cls = JoboServerError if status >= 500 else JoboAPIError
    error: JoboAPIError = error_cls(
        message,
        status_code=status,
        code=code,
        detail=detail,
        body=body,
        retry_after=retry_after,
    )
    if isinstance(error, JoboInsufficientCreditsError) and error.top_up_url:
        error.args = (f"{message} Top up: {error.top_up_url}",)
    return error
