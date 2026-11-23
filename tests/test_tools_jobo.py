"""Unit tests with mocked HTTP. No test touches the network."""

import asyncio
import inspect
import json
import socket
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import httpx
import pytest

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.jobo import (
    JoboAPIError,
    JoboAuthenticationError,
    JoboConnectionError,
    JoboInsufficientCreditsError,
    JoboNotFoundError,
    JoboRateLimitError,
    JoboServerError,
    JoboToolSpec,
    JoboValidationError,
)

KEY = "jbe_live_unit_test_key_0000"
JOB_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

JOB: Dict[str, Any] = {
    "id": JOB_ID,
    "title": "Senior Backend Engineer",
    "company": {"id": "c1", "name": "Acme", "industries": ["Software"]},
    "locations": [{"location": "Berlin, Germany"}],
    "workplace_type": "Remote",
    "employment_type": "Full-time",
    "experience_level": "Senior",
    "compensation": {
        "min": 90000,
        "max": 120000,
        "currency": "USD",
        "period": "yearly",
    },
    "date_posted": "2026-11-01T00:00:00Z",
    "listing_url": "https://jobs.example.com/1",
    "apply_url": "https://jobs.example.com/1/apply",
    "source": "greenhouse",
    "summary": "Build APIs.",
    "description": "Full employer description.",
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
    def body(self) -> Dict[str, Any]:
        return json.loads(self.last.content)


def reply(
    status: int, body: Any = None, headers: Optional[Dict[str, str]] = None
) -> Handler:
    return lambda _request: httpx.Response(status, json=body, headers=headers)


def spec(handler: Handler, **kwargs: Any) -> Tuple[JoboToolSpec, Mock, List[float]]:
    api = Mock(handler)
    tool = JoboToolSpec(
        api_key=KEY,
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


def search_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"jobs": [JOB], "total": 42, "page": 1, "page_size": 5, "total_pages": 9},
        headers={"X-Credits-Deducted": "3", "X-Credits-Balance": "4997"},
    )


@pytest.fixture(autouse=True)
def _isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JOBO_API_KEY", raising=False)

    def blocked(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("network access attempted in a unit test")

    monkeypatch.setattr(socket.socket, "connect", blocked)


# ------------------------------------------------------------------ the spec


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in JoboToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions_match_methods() -> None:
    expected = [
        ("search_jobs", "asearch_jobs"),
        ("get_job", "aget_job"),
        ("find_companies", "afind_companies"),
        ("find_filter_values", "afind_filter_values"),
        ("find_locations", "afind_locations"),
    ]
    assert JoboToolSpec.spec_functions == expected
    for sync_name, async_name in cast(
        List[Tuple[str, str]], JoboToolSpec.spec_functions
    ):
        assert inspect.isfunction(getattr(JoboToolSpec, sync_name))
        assert not inspect.iscoroutinefunction(getattr(JoboToolSpec, sync_name))
        assert inspect.iscoroutinefunction(getattr(JoboToolSpec, async_name))
        # The async twin takes exactly the arguments of the sync tool.
        assert inspect.signature(
            getattr(JoboToolSpec, sync_name)
        ).parameters.keys() == (
            inspect.signature(getattr(JoboToolSpec, async_name)).parameters.keys()
        )


def test_to_tool_list_builds_five_tools_with_billing_descriptions() -> None:
    tool, _, _ = spec(reply(200, {}))
    tools = tool.to_tool_list()
    by_name = {t.metadata.name: t for t in tools}
    assert set(by_name) == {
        "search_jobs",
        "get_job",
        "find_companies",
        "find_filter_values",
        "find_locations",
    }
    assert (
        "billed per job returned"
        in (by_name["search_jobs"].metadata.description or "").lower()
    )
    for free in ("get_job", "find_companies", "find_filter_values", "find_locations"):
        assert "free" in (by_name[free].metadata.description or "").lower()
    assert all(t.async_fn is not None for t in tools)
    props = by_name["search_jobs"].metadata.get_parameters_dict()["properties"]
    assert {"query", "location", "work_model", "page_size", "companies"} <= set(props)


def test_key_comes_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBO_API_KEY", f"  {KEY}  ")
    api = Mock(search_ok)
    JoboToolSpec(http_client=api.client).search_jobs(query="x")
    assert api.last.headers["X-Api-Key"] == KEY


def test_missing_or_blank_key_is_an_error_at_construction() -> None:
    with pytest.raises(ValueError, match="JOBO_API_KEY"):
        JoboToolSpec()
    with pytest.raises(ValueError, match="JOBO_API_KEY"):
        JoboToolSpec(api_key="   ")


def test_bad_max_page_size_rejected() -> None:
    for bad in (0, 101):
        with pytest.raises(ValueError, match="max_page_size"):
            JoboToolSpec(api_key=KEY, max_page_size=bad)


def test_key_is_never_exposed() -> None:
    tool, _, _ = spec(search_ok)
    docs = tool.search_jobs(query="x")
    assert KEY not in repr(tool) and KEY not in repr(tool._client)
    assert KEY not in str(vars(tool).keys())
    assert all(KEY not in d.text and KEY not in json.dumps(d.metadata) for d in docs)


# --------------------------------------------------------------- search_jobs


def test_search_request_shape_and_headers() -> None:
    tool, api, _ = spec(search_ok)
    docs = tool.search_jobs(
        query="backend engineer",
        location=["Berlin, Germany", "Munich, Germany"],
        work_model=["remote", "Hybrid"],
        employment_type="full-time, contract",
        experience_level=["senior"],
        skills="Python, Kubernetes",
        exclude_skills=["PHP"],
        companies=["FedEx, Inc."],
        exclude_companies="Acme Staffing",
        industries=["Software Development"],
        job_categories=["Healthcare"],
        occupation_ids="esco:1,esco:2",
        sources="greenhouse",
        language="en",
        company_categories=["saas"],
        min_salary_usd=90000,
        max_salary_usd=150000,
        posted_after="2026-01-31",
        posted_before="2026-02-01T09:00:00+00:00",
        sort="discovered_desc",
        page=2,
        page_size=3,
    )
    request = api.last
    assert request.method == "POST"
    assert str(request.url) == "https://connect.jobo.world/api/jobs/search"
    assert request.headers["X-Api-Key"] == KEY
    assert request.headers["X-Jobo-Client"] == "llamaindex"
    assert request.headers["User-Agent"].startswith("llama-index-tools-jobo/")
    assert api.body == {
        "queries": ["backend engineer"],
        "locations": ["Berlin, Germany", "Munich, Germany"],
        "sources": ["greenhouse"],
        "work_models": ["remote", "hybrid"],
        "employment_types": ["full-time", "contract"],
        "experience_levels": ["senior"],
        "languages": ["en"],
        "skills": {"include": ["Python", "Kubernetes"], "exclude": ["PHP"]},
        # A name is never split on commas.
        "companies": {"include": ["FedEx, Inc."], "exclude": ["Acme Staffing"]},
        "industries": {"include": ["Software Development"]},
        "company_categories": {"include": ["saas"]},
        "job_categories": {"include": ["Healthcare"]},
        "occupation_ids": {"include": ["esco:1", "esco:2"]},
        "salary_usd": {"min": 90000, "max": 150000},
        "posted_after": "2026-01-31T00:00:00Z",
        "posted_before": "2026-02-01T09:00:00Z",
        "sort": "discovered_desc",
        "page": 2,
        "page_size": 3,
    }
    assert len(docs) == 2  # a summary and one job


def test_search_defaults_send_only_page_and_size() -> None:
    tool, api, _ = spec(search_ok)
    tool.search_jobs()
    assert api.body == {"page": 1, "page_size": 5}


def test_search_documents_carry_jobs_summary_and_usage() -> None:
    tool, _, _ = spec(search_ok)
    summary, job = tool.search_jobs(query="x")
    assert all(isinstance(d, Document) for d in (summary, job))
    assert summary.metadata["type"] == "search_summary"
    assert summary.metadata["total"] == 42
    assert summary.metadata["credits_deducted"] == 3
    assert summary.metadata["credits_balance"] == 4997
    assert "42 matching jobs" in summary.text and "request page 2" in summary.text
    assert job.id_ == JOB_ID
    assert job.metadata["type"] == "job"
    assert job.metadata["url"] == "https://jobs.example.com/1"
    assert "USD 90,000-120,000 per year" in job.text
    assert f"ID: {JOB_ID}" in job.text


def test_search_empty_result_says_so_and_reports_warnings() -> None:
    body = {
        "jobs": [],
        "total": 0,
        "warnings": [{"message": "No company matched: Glenmark."}],
    }
    tool, _, _ = spec(reply(200, body))
    docs = tool.search_jobs(companies=["Glenmark"])
    assert len(docs) == 1
    assert "No jobs matched" in docs[0].text
    assert "No company matched: Glenmark." in docs[0].text


def test_page_size_is_capped_with_a_note() -> None:
    tool, api, _ = spec(search_ok, max_page_size=10)
    summary = tool.search_jobs(page_size=100)[0]
    assert api.body["page_size"] == 10
    assert "capped at 10" in summary.text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"work_model": ["office"]},
        {"employment_type": "permanent"},
        {"experience_level": ["guru"]},
        {"company_categories": ["bank"]},
        {"sort": "newest"},
        {"posted_after": "last week"},
    ],
)
def test_bad_filter_value_is_rejected_before_any_request(
    kwargs: Dict[str, Any],
) -> None:
    tool, api, _ = spec(search_ok)
    with pytest.raises(ValueError):
        tool.search_jobs(**kwargs)
    assert api.requests == []


# ------------------------------------------------------------------- get_job


def test_get_job_request_and_prose() -> None:
    tool, api, _ = spec(reply(200, JOB))
    [doc] = tool.get_job(f"  {JOB_ID.upper()} ")
    assert api.last.method == "GET"
    assert str(api.last.url) == f"https://connect.jobo.world/api/jobs/{JOB_ID}"
    assert api.last.headers["X-Api-Key"] == KEY
    assert doc.id_ == JOB_ID
    assert "# Senior Backend Engineer" in doc.text
    assert "## Description\nFull employer description." in doc.text
    [short] = tool.get_job(JOB_ID, include_description=False)
    assert "Full employer description." not in short.text


def test_get_job_with_a_non_uuid_makes_no_request() -> None:
    tool, api, _ = spec(reply(200, JOB))
    [doc] = tool.get_job("not-an-id")
    assert api.requests == []
    assert "is not a Jobo job id" in doc.text


def test_get_job_unknown_id_is_a_message_but_other_404s_raise() -> None:
    tool, _, _ = spec(reply(404, {"title": "Job not found", "status": 404}))
    [doc] = tool.get_job(JOB_ID)
    assert "No job exists" in doc.text

    tool, _, _ = spec(reply(404, {"error": "no route", "detail": "wrong host"}))
    with pytest.raises(JoboNotFoundError):
        tool.get_job(JOB_ID)


# ------------------------------------------------------------ lookup requests


def test_find_companies_request_and_text() -> None:
    body = {
        "suggestions": [
            {
                "name": "Stripe",
                "domain": "stripe.com",
                "country": "US",
                "job_count": 120,
                "ids": ["id-1", "id-2"],
            }
        ]
    }
    tool, api, _ = spec(reply(200, body))
    [doc] = tool.find_companies("  stripe ", limit=500)
    assert api.last.method == "GET"
    assert api.last.url.path == "/api/connectors/companies/suggest"
    assert dict(api.last.url.params) == {"q": "stripe", "limit": "20"}
    assert api.last.headers["X-Api-Key"] == KEY
    assert "Stripe (stripe.com, US) - 120 jobs live" in doc.text
    assert "company ids: id-1, id-2" in doc.text


def test_find_filter_values_request_and_text() -> None:
    body = {"options": [{"key": "Software Development", "count": 1200}]}
    tool, api, _ = spec(reply(200, body))
    [doc] = tool.find_filter_values("Industries", query=" soft ", limit=5)
    assert api.last.url.path == "/api/connectors/filter-options/industries"
    assert dict(api.last.url.params) == {"limit": "5", "q": "soft"}
    assert "Software Development - 1,200 jobs" in doc.text
    assert doc.metadata == {"type": "filter_values", "facet": "industries"}

    tool.find_filter_values("sources")
    assert dict(api.last.url.params) == {"limit": "20"}


def test_unknown_facet_is_rejected_before_any_request() -> None:
    tool, api, _ = spec(reply(200, {}))
    with pytest.raises(ValueError, match="Unknown facet"):
        tool.find_filter_values("companies")
    assert api.requests == []


TAXONOMY = {
    "categories": [
        {
            "name": "Healthcare",
            "subcategories": [
                {
                    "name": "Nursing",
                    "occupations": [
                        {"id": "esco:nurse", "name": "Registered Nurse"},
                        {"id": "esco:nursery", "name": "Nursery School Teacher"},
                    ],
                }
            ],
        }
    ]
}


def test_occupations_by_name_read_the_taxonomy_once_and_rank_whole_words() -> None:
    tool, api, _ = spec(reply(200, TAXONOMY))
    [doc] = tool.find_filter_values("occupations", query="nurse")
    assert api.last.url.path == "/api/occupations"
    lines = [line for line in doc.text.splitlines() if line.startswith("1.")]
    assert "Registered Nurse - id: esco:nurse (Healthcare > Nursing)" in lines[0]
    tool.find_filter_values("occupations", query="nursery")
    assert len(api.requests) == 1  # cached


def test_find_locations_request_and_text() -> None:
    body = {
        "suggestions": [
            {"display_name": "Springfield, Illinois, United States"},
            {"city": "Springfield", "region": "Missouri", "country": "United States"},
        ]
    }
    tool, api, _ = spec(reply(200, body))
    [doc] = tool.find_locations(" Springfield ", limit=50)
    assert api.last.url.path == "/api/connectors/locations/suggest"
    assert dict(api.last.url.params) == {"q": "Springfield", "limit": "10"}
    assert "1. Springfield, Illinois, United States" in doc.text
    assert "2. Springfield, Missouri, United States" in doc.text


SANDBOX_404 = {"code": "not_available_in_sandbox", "detail": "Not in the sandbox."}


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.find_companies("stripe"),
        lambda t: t.find_filter_values("skills"),
        lambda t: t.find_filter_values("occupations", query="nurse"),
        lambda t: t.find_locations("berlin"),
    ],
)
def test_sandbox_refusal_is_a_message_the_agent_can_act_on(
    call: Callable[..., Any],
) -> None:
    tool, _, _ = spec(reply(404, SANDBOX_404))
    [doc] = call(tool)
    assert "sandbox" in doc.text and "Do not call this tool again" in doc.text
    assert doc.metadata["type"] == "lookup_unavailable"


# ------------------------------------------------------------- error mapping

CALLS: List[Tuple[str, Callable[[JoboToolSpec], Any]]] = [
    ("search_jobs", lambda t: t.search_jobs(query="x")),
    ("get_job", lambda t: t.get_job(JOB_ID)),
    ("find_companies", lambda t: t.find_companies("stripe")),
    ("find_filter_values", lambda t: t.find_filter_values("skills")),
    (
        "find_filter_values_occupations",
        lambda t: t.find_filter_values("occupations", "nurse"),
    ),
    ("find_locations", lambda t: t.find_locations("berlin")),
]
ACALLS: List[Tuple[str, Callable[[JoboToolSpec], Any]]] = [
    ("asearch_jobs", lambda t: t.asearch_jobs(query="x")),
    ("aget_job", lambda t: t.aget_job(JOB_ID)),
    ("afind_companies", lambda t: t.afind_companies("stripe")),
    ("afind_filter_values", lambda t: t.afind_filter_values("skills")),
    (
        "afind_filter_values_occupations",
        lambda t: t.afind_filter_values("occupations", "nurse"),
    ),
    ("afind_locations", lambda t: t.afind_locations("berlin")),
]
STATUSES = [
    (
        401,
        {"error": "invalid_api_key", "detail": "Invalid API key."},
        JoboAuthenticationError,
        None,
    ),
    (
        402,
        {
            "code": "insufficient_credits",
            "detail": "Insufficient credits.",
            "top_up_url": "https://enterprise.jobo.world/top-up",
        },
        JoboInsufficientCreditsError,
        None,
    ),
    (
        429,
        {"code": "rate_limit_exceeded", "detail": "Too many requests."},
        JoboRateLimitError,
        {"Retry-After": "7"},
    ),
]


@pytest.mark.parametrize(("status", "body", "error", "headers"), STATUSES)
@pytest.mark.parametrize(("name", "call"), CALLS, ids=[c[0] for c in CALLS])
def test_api_errors_map_to_typed_exceptions(
    name: str,
    call: Callable[[JoboToolSpec], Any],
    status: int,
    body: Dict[str, Any],
    error: type,
    headers: Optional[Dict[str, str]],
) -> None:
    # Retry-After 7s is inside the wait limit, so disable retries to see the 429.
    tool, api, sleeps = spec(reply(status, body, headers), max_retries=0)
    with pytest.raises(error) as caught:
        call(tool)
    assert caught.value.status_code == status  # type: ignore[attr-defined]
    assert len(api.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize(("status", "body", "error", "headers"), STATUSES)
@pytest.mark.parametrize(("name", "call"), ACALLS, ids=[c[0] for c in ACALLS])
def test_async_api_errors_map_to_typed_exceptions(
    name: str,
    call: Callable[[JoboToolSpec], Any],
    status: int,
    body: Dict[str, Any],
    error: type,
    headers: Optional[Dict[str, str]],
) -> None:
    tool, api, _ = spec(reply(status, body, headers), max_retries=0)
    with pytest.raises(error):
        asyncio.run(call(tool))
    assert len(api.requests) == 1


def test_402_message_has_top_up_link_and_is_never_retried() -> None:
    body = {"code": "insufficient_credits", "top_up_url": "https://x.example/top-up"}
    tool, api, sleeps = spec(reply(402, body))  # default max_retries=2
    with pytest.raises(
        JoboInsufficientCreditsError, match="Top up: https://x.example/top-up"
    ):
        tool.search_jobs(query="x")
    assert len(api.requests) == 1 and sleeps == []


def test_401_is_never_retried() -> None:
    tool, api, _ = spec(reply(401, {"error": "invalid_api_key"}))
    with pytest.raises(JoboAuthenticationError):
        tool.get_job(JOB_ID)
    assert len(api.requests) == 1


def test_429_is_retried_honouring_retry_after_then_raised() -> None:
    headers = {"Retry-After": "7"}
    tool, api, sleeps = spec(reply(429, {"code": "rate_limit_exceeded"}, headers))
    with pytest.raises(JoboRateLimitError, match="Retry after 7 seconds") as caught:
        tool.find_locations("berlin")
    assert caught.value.retry_after == 7.0
    assert len(api.requests) == 3  # first attempt plus two retries
    assert sleeps == [7.0, 7.0]


def test_a_retry_after_beyond_the_wait_limit_is_raised_not_slept() -> None:
    tool, api, sleeps = spec(reply(429, {"code": "x"}, {"Retry-After": "600"}))
    with pytest.raises(JoboRateLimitError) as caught:
        tool.search_jobs(query="x")
    assert caught.value.retry_after == 600.0
    assert len(api.requests) == 1 and sleeps == []


def test_429_then_success_recovers() -> None:
    first = httpx.Response(429, json={}, headers={"Retry-After": "1"})
    answers = iter([first, search_ok(httpx.Request("POST", "https://x.example"))])
    tool, api, sleeps = spec(lambda _r: next(answers))
    assert len(tool.search_jobs(query="x")) == 2
    assert len(api.requests) == 2 and sleeps == [1.0]


def test_5xx_and_400_mapping() -> None:
    tool, api, _ = spec(reply(503, {"message": "down"}))
    with pytest.raises(JoboServerError):
        tool.search_jobs(query="x")
    assert len(api.requests) == 3

    problem = {
        "title": "One or more validation errors occurred.",
        "errors": {"page_size": ["page_size must be between 1 and 100."]},
    }
    tool, api, _ = spec(reply(400, problem))
    with pytest.raises(
        JoboValidationError, match="page_size: page_size must be between"
    ):
        tool.search_jobs(query="x")
    assert len(api.requests) == 1

    tool, _, _ = spec(reply(418, {"detail": "teapot"}))
    with pytest.raises(JoboAPIError) as caught:
        tool.search_jobs(query="x")
    assert caught.value.status_code == 418


def test_read_timeout_is_not_retried_but_connect_failure_is() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    tool, api, _ = spec(timeout)
    with pytest.raises(JoboConnectionError):
        tool.search_jobs(query="x")
    assert len(api.requests) == 1  # a billed page may have been delivered

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    tool, api, _ = spec(refused)
    with pytest.raises(JoboConnectionError):
        tool.search_jobs(query="x")
    assert len(api.requests) == 3


# ---------------------------------------------------------------------- async


def test_async_search_and_lookups_use_the_same_requests() -> None:
    tool, api, _ = spec(search_ok, max_page_size=2)
    docs = asyncio.run(
        tool.asearch_jobs(query="nurse", work_model="remote", page_size=9)
    )
    assert api.body == {
        "queries": ["nurse"],
        "work_models": ["remote"],
        "page": 1,
        "page_size": 2,
    }
    assert api.last.headers["X-Api-Key"] == KEY
    assert "capped at 2" in docs[0].text

    tool, api, _ = spec(reply(200, JOB))
    [doc] = asyncio.run(tool.aget_job(JOB_ID))
    assert str(api.last.url).endswith(f"/api/jobs/{JOB_ID}") and doc.id_ == JOB_ID

    tool, api, _ = spec(reply(200, {"suggestions": []}))
    asyncio.run(tool.afind_companies("stripe"))
    assert api.last.url.path == "/api/connectors/companies/suggest"
    asyncio.run(tool.afind_locations("berlin"))
    assert api.last.url.path == "/api/connectors/locations/suggest"

    tool, api, _ = spec(reply(200, {"options": []}))
    asyncio.run(tool.afind_filter_values("skills", "py"))
    assert api.last.url.path == "/api/connectors/filter-options/skills"


def test_agent_tool_call_goes_through_the_function_tool() -> None:
    tool, api, _ = spec(search_ok)
    search = {t.metadata.name: t for t in tool.to_tool_list()}["search_jobs"]
    output = search.call(query="python", page_size=1)
    assert api.body["queries"] == ["python"]
    assert "Senior Backend Engineer" in str(output)
