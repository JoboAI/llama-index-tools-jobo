"""Hiring signals: the request, the text, the cost and the opt-in. Mocked HTTP."""

import asyncio
import inspect
import socket
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import pytest

from llama_index.tools.jobo import (
    JoboConnectionError,
    JoboInsufficientCreditsError,
    JoboNotFoundError,
    JoboPermissionError,
    JoboRateLimitError,
    JoboServerError,
    JoboToolSpec,
    JoboValidationError,
)
from llama_index.tools.jobo._signals import (
    NO_SIGNALS_IN_SANDBOX,
    build_signals_request,
    describe_signal,
    signal_strength_band,
    signal_type_label,
)

KEY = "jbe_live_unit_test_key_0000"
SANDBOX_KEY = "jbe_test_unit_test_key_0000"
COMPANY_ID = "6c1f0a9e-3b7d-4c55-8e21-5a4d9f0b7c33"
OTHER_COMPANY_ID = "11111111-1111-4111-8111-111111111111"
EVIDENCE = [
    "a41c7e02-58b1-4d0f-b6c9-0e2f7d5a1c88",
    "00000000-0000-4000-8000-000000000042",
]
CURSOR = "eyJkIjoiMjAyNi0xMS0wOSJ9"

SURGE: Dict[str, Any] = {
    "id": "0d0f5c1e-7a52-4f0e-9d3b-2f6f1c7a9b10",
    "type": "hiring_surge",
    "detected_on": "2026-11-10",
    "created_at": "2026-11-11T02:14:09.1234567Z",
    "window_start": "2026-11-04",
    "window_end": "2026-11-10",
    "strength": 88,
    "company": {
        "id": COMPANY_ID,
        "name": "Northwind Health",
        "domain": "northwind.example",
        "country_code": "US",
        "is_agency": False,
    },
    "details": {"new_jobs": 42, "baseline_weekly_avg": 9, "ratio": 4.7},
    "focus": {"functions": ["healthcare"], "topics": [], "job_countries": ["US"]},
    "evidence_job_ids": EVIDENCE,
}

#: A type added to the API after this package was written.
NEW_TYPE: Dict[str, Any] = {
    "id": "2a6f3c1e-0000-4f0e-9d3b-2f6f1c7a9b11",
    "type": "remote_shift",
    "detected_on": "2026-11-09",
    "strength": 12,
    "company": {"id": OTHER_COMPANY_ID, "name": "Globex", "domain": None},
    "details": {"share": 0.4},
    "evidence_job_ids": [],
}

PAGE: Dict[str, Any] = {
    "signals": [SURGE, NEW_TYPE],
    "next_cursor": CURSOR,
    "has_more": True,
}

#: A Job Search plan whose included signals are used up: the wallet pays.
METERING = {
    "X-Credits-Deducted": "60",
    "X-Credits-Balance": "4940",
    "X-Quota-Limit": "250",
    "X-Quota-Remaining": "0",
}

#: What the model reads from `PAGE`: each Document's text, one after another.
PAGE_TEXT = "\n".join(
    [
        "2 hiring signals, newest detection day first. More match: call again "
        f'with the same arguments and cursor "{CURSOR}" to read the next page.',
        "1. Northwind Health (northwind.example) - Hiring surge",
        "   42 new jobs this week vs 9 a week normally",
        "   Detected: 2026-11-10 | Strength: Very strong (88/100)",
        f"   Company ID: {COMPANY_ID}",
        f"   Evidence jobs: {EVIDENCE[0]}, {EVIDENCE[1]}",
        "2. Globex - Remote shift",
        "   Detected: 2026-11-09 | Strength: Mild (12/100)",
        f"   Company ID: {OTHER_COMPANY_ID}",
        "Evidence jobs are the postings behind a signal: read one free with get_job.",
        "Billed: 2 hiring signals (0 of 250 included signals left this month; 60 "
        "credits charged to the wallet; wallet balance 4,940 credits). Reading "
        "the same signals again is billed again.",
    ]
)

SANDBOX_404 = {"code": "not_available_in_sandbox", "detail": "Not in the sandbox."}
ACCOUNT_KEY_REQUIRED = {
    "title": "Account API key required",
    "status": 403,
    "detail": "Hiring Signals is charged per signal to an account's wallet.",
    "code": "account_key_required",
}

Handler = Callable[[httpx.Request], httpx.Response]


class Mock:
    """A mock Jobo API: records every request and answers with `handler`."""

    def __init__(self, handler: Handler) -> None:
        self.requests: List[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        transport = httpx.MockTransport(record)
        self.client = httpx.Client(transport=transport)
        self.async_client = httpx.AsyncClient(transport=transport)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    @property
    def params(self) -> Dict[str, str]:
        return dict(self.last.url.params)


def reply(
    status: int, body: Any = None, headers: Optional[Dict[str, str]] = None
) -> Handler:
    return lambda _request: httpx.Response(status, json=body, headers=headers)


def page(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=PAGE, headers=METERING)


def lost(error: Callable[..., Exception]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("the response was lost", request=request)

    return handler


def spec(handler: Handler, **kwargs: Any) -> Tuple[JoboToolSpec, Mock, List[float]]:
    api = Mock(handler)
    tool = JoboToolSpec(
        api_key=kwargs.pop("api_key", KEY),
        http_client=api.client,
        http_async_client=api.async_client,
        **kwargs,
    )
    sleeps: List[float] = []

    async def asleep(seconds: float) -> None:
        sleeps.append(seconds)

    tool._client._sleep = sleeps.append  # type: ignore[assignment]
    tool._client._asleep = asleep  # type: ignore[assignment]
    return tool, api, sleeps


@pytest.fixture(autouse=True)
def _isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JOBO_API_KEY", raising=False)

    def blocked(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("network access attempted in a unit test")

    monkeypatch.setattr(socket.socket, "connect", blocked)


# -------------------------------------------------------------------- opt-in


def test_hiring_signals_are_not_in_the_default_tool_list() -> None:
    tool, _, _ = spec(page)

    names = {t.metadata.name for t in tool.to_tool_list()}
    default = JoboToolSpec.spec_functions

    assert "get_hiring_signals" not in names
    assert len(names) == 5
    assert ("get_hiring_signals", "aget_hiring_signals") not in default


def test_include_hiring_signals_adds_the_tool_with_its_async_twin() -> None:
    tool, _, _ = spec(page, include_hiring_signals=True)

    tools = {t.metadata.name: t for t in tool.to_tool_list()}
    signals = tools["get_hiring_signals"]
    # The description is the signature and the docstring, line breaks and all.
    description = " ".join((signals.metadata.description or "").lower().split())

    assert len(tools) == 6
    assert signals.async_fn is not None
    assert "billed per signal returned" in description
    assert "ten times the price of a job" in description
    assert "a signal read again is billed again" in description
    assert "keep limit small" in description
    assert "returns no signals costs nothing" in description
    props = signals.metadata.get_parameters_dict()["properties"]
    assert {"types", "domains", "company_id", "limit", "cursor"} <= set(props)
    # The class, and every other spec, keeps the default list.
    assert len(JoboToolSpec.spec_functions) == 5
    other, _, _ = spec(page)
    assert len(other.to_tool_list()) == 5


def test_the_async_twin_takes_exactly_the_arguments_of_the_sync_tool() -> None:
    sync = inspect.signature(JoboToolSpec.get_hiring_signals).parameters
    asynchronous = inspect.signature(JoboToolSpec.aget_hiring_signals).parameters

    assert sync.keys() == asynchronous.keys()
    assert inspect.iscoroutinefunction(JoboToolSpec.aget_hiring_signals)


def test_bad_max_signals_rejected() -> None:
    for bad in (0, 201):
        with pytest.raises(ValueError, match="max_signals"):
            JoboToolSpec(api_key=KEY, max_signals=bad)


# ------------------------------------------------------------------- request


def test_the_request_joins_each_list_with_commas() -> None:
    tool, api, _ = spec(page)

    tool.get_hiring_signals(
        types="hiring_surge, New_Country",
        domains=["stripe.com", " shopify.com ", "stripe.com"],
        countries="DE,AT",
        company_sizes=["51-200", "201-500"],
        industries="Financial Services",
        funding_stages=["Seed", "Series A"],
        functions=["Sales"],
        roles="account executive",
        seniority=["Director", "vp"],
        technologies="Salesforce, HubSpot",
        skills=["Negotiation"],
        job_countries=["DE"],
        min_strength=60,
        detected_after="2026-11-01",
        detected_before="2026-11-23",
        created_after="2026-11-20T08:00:00.1234567Z",
        exclude_agencies=True,
        limit=3,
        cursor="abc",
    )

    assert api.last.method == "GET"
    assert str(api.last.url).startswith("https://connect.jobo.world/api/signals?")
    assert api.last.headers["X-Api-Key"] == KEY
    assert api.last.headers["X-Jobo-Client"] == "llamaindex"
    assert api.params == {
        "types": "hiring_surge,new_country",
        "domains": "stripe.com,shopify.com",
        "countries": "DE,AT",
        "company_sizes": "51-200,201-500",
        "industries": "Financial Services",
        "funding_stages": "Seed,Series A",
        "exclude_agencies": "true",
        "functions": "Sales",
        "roles": "account executive",
        "seniority": "director,vp",
        "technologies": "Salesforce,HubSpot",
        "skills": "Negotiation",
        "job_countries": "DE",
        "min_strength": "60",
        "detected_after": "2026-11-01",
        "detected_before": "2026-11-23",
        # Sent as written: `created_after` is exclusive, and a parsed instant
        # would lose the seventh digit and read the same signal again.
        "created_after": "2026-11-20T08:00:00.1234567Z",
        "limit": "3",
        "cursor": "abc",
    }


def test_an_empty_call_asks_for_five_signals_and_nothing_else() -> None:
    tool, api, _ = spec(page)

    tool.get_hiring_signals(exclude_agencies=False, domains=[" "], cursor=" ")

    assert api.params == {"limit": "5"}


def test_limit_is_capped_with_a_note() -> None:
    tool, api, _ = spec(page, max_signals=10)

    summary = tool.get_hiring_signals(limit=100)[0]

    assert api.params["limit"] == "10"
    assert "Note: limit was capped at 10" in summary.text


def test_company_id_reads_one_company_from_its_own_endpoint() -> None:
    tool, api, _ = spec(page)

    tool.get_hiring_signals(
        company_id=f" {COMPANY_ID.upper()} ",
        types=["hiring_freeze", "hiring_slowdown"],
        min_strength=50,
        detected_after="2026-10-01",
        exclude_agencies=False,
    )

    assert api.last.url.path == f"/api/companies/{COMPANY_ID}/signals"
    assert api.params == {
        "types": "hiring_freeze,hiring_slowdown",
        "min_strength": "50",
        "detected_after": "2026-10-01",
        "limit": "5",
    }


def test_company_id_refuses_the_filters_its_endpoint_ignores() -> None:
    tool, api, _ = spec(page)

    with pytest.raises(ValueError, match="Leave out countries, functions,"):
        tool.get_hiring_signals(
            company_id=COMPANY_ID,
            types="hiring_surge",
            countries="DE",
            functions="Sales",
        )

    assert api.requests == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"company_id": "acme.com"}, "is not a Jobo company id"),
        (
            {"industries": ["Film, TV & Stage Production"]},
            "cannot be used in industries",
        ),
        # A name is one value, so a comma in it is refused, not split.
        ({"funding_stages": "Series A, Series B"}, "cannot be used in funding_stages"),
        (
            {"domains": [f"company{index}.example" for index in range(101)]},
            "domains takes at most 100 values",
        ),
        ({"types": ["remote_shift"]}, "Unknown types"),
        ({"seniority": "intern"}, "Unknown seniority"),
        ({"company_sizes": ["1-5"]}, "Unknown company_sizes"),
        ({"min_strength": 0}, "min_strength must be 1-100"),
        ({"min_strength": 101}, "min_strength must be 1-100"),
        ({"detected_after": "last week"}, "detected_after must be a day"),
        ({"detected_before": "2026-11-1"}, "detected_before must be a day"),
        ({"created_after": "2026-11-20"}, "created_after must be an ISO 8601"),
    ],
)
def test_bad_values_are_rejected_before_any_request(
    kwargs: Dict[str, Any], message: str
) -> None:
    tool, api, _ = spec(page)

    with pytest.raises(ValueError, match=message):
        tool.get_hiring_signals(**kwargs)
    with pytest.raises(ValueError, match=message):
        asyncio.run(tool.aget_hiring_signals(**kwargs))

    assert api.requests == []


def test_build_signals_request_takes_every_filter_of_the_api() -> None:
    """Also the ones the tool leaves to the developer."""
    path, params = build_signals_request(
        {
            "exclude_domains": "acme.com",
            "categories": ["SaaS"],
            "founded_after": 2018,
            "job_cities": ["Austin"],
            "min_days_open": 60.0,
            "reposted_only": True,
            "min_new_jobs": 0,
        },
        limit=200,
        cursor=" next ",
    )

    assert path == "/api/signals"
    assert params == {
        "exclude_domains": "acme.com",
        "categories": "SaaS",
        "founded_after": 2018,
        "job_cities": "Austin",
        "min_days_open": 60,
        "reposted_only": True,
        "limit": 200,
        "cursor": "next",
    }
    with pytest.raises(ValueError, match="Unknown hiring signal filter"):
        build_signals_request({"company": "acme.com"})
    with pytest.raises(ValueError, match="limit must be between 1 and 200"):
        build_signals_request(limit=201)


# ---------------------------------------------------------------------- text


def test_the_model_reads_one_block_per_signal_and_the_cost() -> None:
    tool, api, _ = spec(page, include_hiring_signals=True)
    signals = {t.metadata.name: t for t in tool.to_tool_list()}["get_hiring_signals"]

    output = signals.call(types=["hiring_surge"], limit=2)

    assert api.params == {"types": "hiring_surge", "limit": "2"}
    assert output.content == PAGE_TEXT
    assert "Document(" not in output.content

    output = asyncio.run(signals.acall(types=["hiring_surge"], limit=2))
    assert output.content == PAGE_TEXT


def test_documents_carry_the_signals_the_cursor_and_the_usage() -> None:
    tool, _, _ = spec(page)

    summary, surge, unknown, billing = tool.get_hiring_signals()

    assert summary.metadata == {
        "type": "signals_summary",
        "signals": 2,
        "has_more": True,
        "next_cursor": CURSOR,
        "credits_deducted": 60,
        "credits_balance": 4940,
        "quota_limit": 250,
        "quota_remaining": 0,
    }
    assert surge.id_ == SURGE["id"]
    assert surge.metadata == {
        "type": "hiring_signal",
        "signal_id": SURGE["id"],
        "signal_type": "hiring_surge",
        "label": "Hiring surge",
        "summary": "42 new jobs this week vs 9 a week normally",
        "company": "Northwind Health",
        "company_id": COMPANY_ID,
        "domain": "northwind.example",
        "detected_on": "2026-11-10",
        "created_at": "2026-11-11T02:14:09.1234567Z",
        "strength": 88,
        "evidence_job_ids": ", ".join(EVIDENCE),
    }
    # A type this version does not know is named, not failed on.
    assert unknown.metadata["label"] == "Remote shift"
    assert billing.metadata["type"] == "signals_billing"
    assert billing.metadata["billed_signals"] == 2
    assert billing.text.endswith("Reading the same signals again is billed again.")


def test_an_empty_page_says_it_cost_nothing() -> None:
    tool, _, _ = spec(reply(200, {"signals": [], "has_more": False}))

    summary, billing = tool.get_hiring_signals(domains="nobody.example")

    free = "Billed: nothing. A request that returns no signals is free."

    assert summary.text.startswith("No hiring signals matched.")
    assert summary.metadata == {
        "type": "signals_summary",
        "signals": 0,
        "has_more": False,
    }
    assert billing.text == free


def test_without_metering_headers_the_page_says_no_charge_was_reported() -> None:
    body = {"signals": [SURGE], "has_more": False}
    tool, _, _ = spec(reply(200, body))

    docs = tool.get_hiring_signals()
    header = docs[0].text

    assert header == "1 hiring signal, newest detection day first. No more match."
    assert docs[-1].text.endswith(
        "Billed: 1 hiring signal. The API reported no charge, as on the "
        "Unlimited plan, which has no per-signal charge."
    )


# -------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("handler", "error"),
    [
        (reply(500, {"message": "boom"}), JoboServerError),
        (lost(httpx.ReadTimeout), JoboConnectionError),
        (lost(httpx.RemoteProtocolError), JoboConnectionError),
    ],
)
def test_a_failure_after_the_request_was_sent_is_not_replayed(
    handler: Handler, error: type
) -> None:
    # The page may have been billed: a replay would bill it again.
    tool, api, sleeps = spec(handler)
    with pytest.raises(error):
        tool.get_hiring_signals(domains="stripe.com")
    with pytest.raises(error):
        asyncio.run(tool.aget_hiring_signals(company_id=COMPANY_ID))
    assert len(api.requests) == 2 and sleeps == []


def test_a_refusal_before_the_charge_is_retried() -> None:
    limited = httpx.Response(429, json={}, headers={"Retry-After": "2"})
    busy = httpx.Response(503, json={}, headers={"Retry-After": "5"})
    answers = iter([limited, busy, page(httpx.Request("GET", "https://x.example"))])
    tool, api, sleeps = spec(lambda _r: next(answers))

    docs = tool.get_hiring_signals()

    assert len(docs) == 4
    assert len(api.requests) == 3 and sleeps == [2.0, 5.0]


def test_a_500_on_a_companys_jobs_is_still_retried() -> None:
    tool, api, sleeps = spec(reply(500, {"message": "boom"}))
    with pytest.raises(JoboServerError):
        tool._client.request("GET", f"/api/companies/{COMPANY_ID}/jobs")
    assert len(api.requests) == 3 and len(sleeps) == 2


@pytest.mark.parametrize(
    ("status", "body", "error", "advice"),
    [
        (
            402,
            {"code": "insufficient_credits", "top_up_url": "https://x.example/t"},
            JoboInsufficientCreditsError,
            "Ask for fewer signals with a smaller limit",
        ),
        (
            403,
            ACCOUNT_KEY_REQUIRED,
            JoboPermissionError,
            "an account API key is needed",
        ),
        (403, {"title": "Forbidden"}, JoboPermissionError, "may not read hiring"),
        (429, {"code": "rate_limit_exceeded"}, JoboRateLimitError, "Wait before"),
        (
            400,
            {"errors": {"roles": ["'a' is too short to look for in a job title."]}},
            JoboValidationError,
            "Correct the arguments named above",
        ),
    ],
)
def test_errors_tell_the_model_what_to_do(
    status: int, body: Dict[str, Any], error: type, advice: str
) -> None:
    tool, api, _ = spec(reply(status, body, {"Retry-After": "7"}), max_retries=0)
    with pytest.raises(error, match=advice) as caught:
        tool.get_hiring_signals()
    assert caught.value.status_code == status  # type: ignore[attr-defined]
    with pytest.raises(error, match=advice):
        asyncio.run(tool.aget_hiring_signals())
    assert len(api.requests) == 2


def test_402_keeps_the_top_up_link() -> None:
    body = {"code": "insufficient_credits", "top_up_url": "https://x.example/top-up"}
    tool, _, _ = spec(reply(402, body))
    with pytest.raises(JoboInsufficientCreditsError, match="https://x.example/top-up"):
        tool.get_hiring_signals()


def test_a_sandbox_key_is_answered_with_a_message_not_an_error() -> None:
    tool, api, _ = spec(reply(404, SANDBOX_404), api_key=SANDBOX_KEY)

    [doc] = tool.get_hiring_signals(types="hiring_surge")
    [again] = asyncio.run(tool.aget_hiring_signals())

    assert doc.text == NO_SIGNALS_IN_SANDBOX
    assert "Do not call this tool again" in doc.text
    assert doc.metadata == {"type": "signals_unavailable"}
    assert again.text == NO_SIGNALS_IN_SANDBOX
    assert api.last.headers["X-Api-Key"] == SANDBOX_KEY


def test_any_other_404_is_an_error() -> None:
    tool, _, _ = spec(reply(404, {"title": "Not Found"}))
    with pytest.raises(JoboNotFoundError):
        tool.get_hiring_signals(company_id=COMPANY_ID)


# ----------------------------------------------------------------- sentences

#: The connector core's table (signal-summary.test.ts): the sentences Jobo's
#: dashboard and alert emails write.
SENTENCES: List[Tuple[str, Dict[str, Any], str]] = [
    (
        "hiring_surge",
        {"new_jobs": 17, "baseline_weekly_avg": 3.1},
        "17 new jobs this week vs 3.1 a week normally",
    ),
    (
        "hiring_surge",
        {"new_jobs": 1, "baseline_weekly_avg": 0.04},
        "1 new job this week vs 0 a week normally",
    ),
    (
        "hiring_surge",
        {"new_jobs": 1200, "baseline_weekly_avg": 1050.25},
        "1,200 new jobs this week vs 1,050.3 a week normally",
    ),
    (
        "hiring_slowdown",
        {
            "open_roles": 40,
            "open_roles_before": 90,
            "change_pct": -55.6,
            "compared_to": "2026-09-21",
        },
        "Open roles down 55.6% since 2026-09-21: 40 now, 90 then",
    ),
    (
        "new_function",
        {"category": "Sales", "new_jobs": 4, "lookback_days": 180},
        "4 new jobs in Sales, a first in six months",
    ),
    (
        "new_function",
        {"category": "Sales", "lookback_days": 90},
        "First Sales job in 3 months",
    ),
    (
        "new_country",
        {"country": "Germany", "new_jobs": 1},
        "First job in Germany in six months",
    ),
    (
        "new_country",
        {"country": "Germany", "new_jobs": 3, "lookback_days": 45},
        "3 new jobs in Germany, a first in 45 days",
    ),
    (
        "new_skill",
        {"skill": "Rust", "new_jobs": 5},
        "5 new jobs ask for Rust, a first in six months",
    ),
    (
        "new_skill",
        {"skill": "Rust", "named": True},
        "First job asking for Rust in six months",
    ),
    (
        "leadership_hire",
        {"title": "VP of Engineering"},
        "Hiring a leader: VP of Engineering",
    ),
    ("first_role", {"occupation": "Data Engineer"}, "First Data Engineer role"),
    (
        "new_technology",
        {"technology": "Snowflake", "title": "Data Engineer"},
        "First mention of Snowflake, in a Data Engineer role",
    ),
    ("new_technology", {"technology": "Snowflake"}, "First mention of Snowflake"),
    (
        "function_surge",
        {
            "category": "Technology",
            "new_jobs": 12,
            "baseline_avg": 2.5,
            "window_days": 14,
        },
        "12 new Technology jobs in 14 days vs 2.5 normally",
    ),
    (
        "function_surge",
        {"category": "Technology", "new_jobs": 1, "baseline_avg": 0.2},
        "1 new Technology job in two weeks vs 0.2 normally",
    ),
    (
        "new_city",
        {"city": "Austin", "region": "Texas", "new_jobs": 6},
        "6 new jobs in Austin, Texas, a first there",
    ),
    ("new_city", {"city": "Austin"}, "First job in Austin"),
    (
        "hard_to_fill",
        {"title": "Staff Nurse", "days_open": 92, "city": "Leeds"},
        "Staff Nurse (Leeds) open 92 days",
    ),
    (
        "hard_to_fill",
        {
            "title": "Staff Nurse",
            "days_open": 30,
            "reason": "reposted",
            "closed_days": 1,
        },
        "Staff Nurse re-posted after 1 day closed",
    ),
    (
        "hard_to_fill",
        {"title": "Staff Nurse", "days_open": 30, "reason": "reposted"},
        "Staff Nurse re-posted",
    ),
    (
        "leadership_role_closed",
        {"title": "Chief Financial Officer", "days_open": 61},
        "Chief Financial Officer search closed after 61 days",
    ),
    (
        "hiring_freeze",
        {"closed_jobs": 45, "open_roles_before": 52, "closed_to": "2026-11-15"},
        "45 of 52 open roles removed in the week to 2026-11-15",
    ),
    ("hiring_resumed", {"gap_days": 74}, "First new job in 74 days"),
]


@pytest.mark.parametrize(("signal_type", "details", "sentence"), SENTENCES)
def test_one_sentence_per_signal(
    signal_type: str, details: Dict[str, Any], sentence: str
) -> None:
    assert describe_signal({"type": signal_type, "details": details}) == sentence


def test_the_sentence_table_covers_every_type() -> None:
    assert len({signal_type for signal_type, _, _ in SENTENCES}) == 14


def test_the_sentence_falls_back_to_the_label() -> None:
    surge = {"type": "hiring_surge", "details": {"new_jobs": 17}}
    leader: Dict[str, Any] = {"type": "leadership_hire", "details": {}}
    city = {"type": "new_city", "details": None}
    # A number must be a number: true is not 1.
    resumed = {"type": "hiring_resumed", "details": {"gap_days": True}}
    unknown = {"type": "remote_shift", "details": {"share": 0.4}}

    assert describe_signal(surge) == "Hiring surge"
    assert describe_signal(leader) == "Leadership hire"
    assert describe_signal(city) == "New city"
    assert describe_signal(resumed) == "Hiring resumed"
    assert describe_signal(unknown) == "Remote shift"
    assert signal_type_label("remote_shift") == "Remote shift"
    assert signal_type_label("") == "Signal"
    assert signal_type_label(None) == "Signal"


def test_strength_reads_as_one_of_four_bands() -> None:
    bands = [signal_strength_band(value) for value in (100, 75, 74, 50, 49, 25, 24, 1)]

    assert bands == [
        "Very strong",
        "Very strong",
        "Strong",
        "Strong",
        "Moderate",
        "Moderate",
        "Mild",
        "Mild",
    ]
