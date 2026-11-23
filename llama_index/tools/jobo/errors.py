"""
Exceptions raised by the Jobo tool spec.

Every non-2xx response becomes a `JoboAPIError` subclass chosen by status code.
"""

from typing import Any, Dict, List, Optional


class JoboError(Exception):
    """Base class for everything this package raises."""


class JoboConnectionError(JoboError):
    """The request never produced an HTTP response (DNS, TLS, socket, timeout)."""


class JoboAPIError(JoboError):
    """
    A non-2xx response from the Jobo API.

    Attributes:
        status_code: HTTP status of the response.
        code: Stable machine-readable error code, when the API sent one.
        detail: Human-readable explanation from the response body, when present.
        body: The parsed response body (a dict for JSON, otherwise text).
        retry_after: Seconds the API asked the client to wait, when it said so.

    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: Optional[str] = None,
        detail: Optional[str] = None,
        body: Any = None,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.body = body
        self.retry_after = retry_after


class JoboValidationError(JoboAPIError):
    """HTTP 400: the request is invalid."""


class JoboAuthenticationError(JoboAPIError):
    """HTTP 401: the API key is missing, invalid, revoked or expired."""


class JoboInsufficientCreditsError(JoboAPIError):
    """
    HTTP 402: included jobs plus wallet balance cannot cover the request.

    Never retried. The API prices a full page before the request runs, so the
    same request fails the same way until the wallet is topped up or a smaller
    `page_size` is sent.
    """

    @property
    def top_up_url(self) -> Optional[str]:
        """Link that opens a wallet top-up covering the shortfall, when sent."""
        if isinstance(self.body, dict):
            value = self.body.get("top_up_url")
            if isinstance(value, str) and value:
                return value
        return None


class JoboPermissionError(JoboAPIError):
    """HTTP 403: this key or account may not use the endpoint."""


class JoboNotFoundError(JoboAPIError):
    """HTTP 404: unknown id, or an endpoint a sandbox key cannot call."""

    @property
    def not_available_in_sandbox(self) -> bool:
        """Whether a sandbox key called an endpoint the sandbox does not serve."""
        return self.code == "not_available_in_sandbox"


class JoboRateLimitError(JoboAPIError):
    """HTTP 429: a rate-limit window is exhausted. `retry_after` has the seconds."""


class JoboServerError(JoboAPIError):
    """HTTP 5xx: the API failed or is temporarily unavailable."""


def field_errors(body: Any) -> Dict[str, List[str]]:
    """The `errors` map of a validation problem: field name to its messages."""
    raw = body.get("errors") if isinstance(body, dict) else None
    if not isinstance(raw, dict):
        return {}
    errors: Dict[str, List[str]] = {}
    for field, messages in raw.items():
        if isinstance(messages, str):
            messages = [messages]
        if not isinstance(messages, list):
            continue
        texts = [m.strip() for m in messages if isinstance(m, str) and m.strip()]
        if texts:
            errors[str(field)] = texts
    return errors
