"""Jobo tool spec."""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote
from uuid import UUID

import httpx

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.jobo._client import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    JoboClient,
    Reply,
)
from llama_index.tools.jobo._format import (
    EMPTY_SEARCH,
    flatten_taxonomy,
    format_company_suggestions,
    format_filter_values,
    format_job_document,
    format_job_summary,
    format_location_suggestions,
    format_occupation_matches,
    job_metadata,
    match_occupations,
    search_header,
    search_notes,
)
from llama_index.tools.jobo._signals import (
    COMPANY_SIZES,
    NO_SIGNALS_IN_SANDBOX,
    SIGNAL_MAX_LIMIT,
    SIGNAL_SENIORITY,
    SIGNAL_TYPES,
    build_signals_request,
    check_day,
    check_enum,
    check_instant,
    format_signal,
    next_cursor,
    signal_list,
    signal_metadata,
    signals_error_advice,
    signals_footer,
    signals_header,
)
from llama_index.tools.jobo.errors import JoboAPIError, JoboNotFoundError

API_MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 5
DEFAULT_MAX_PAGE_SIZE = 25
#: Signals `get_hiring_signals` asks for when the model names no limit.
DEFAULT_SIGNAL_LIMIT = 5
DEFAULT_MAX_SIGNALS = 25
#: The billed hiring signals tool, sync and async. Not in `spec_functions`:
#: `JoboToolSpec(include_hiring_signals=True)` adds it.
SIGNALS_SPEC_FUNCTION = ("get_hiring_signals", "aget_hiring_signals")
_TAXONOMY_TTL_SECONDS = 600.0

WORK_MODELS = ("remote", "hybrid", "onsite")
EMPLOYMENT_TYPES = (
    "full-time",
    "part-time",
    "contract",
    "internship",
    "freelance",
    "temporary",
)
EXPERIENCE_LEVELS = ("intern", "entry", "mid", "senior", "lead", "executive")
COMPANY_CATEGORIES = (
    "b2b",
    "b2c",
    "b2b2c",
    "b2g",
    "saas",
    "marketplace",
    "e-commerce",
    "service-provider",
    "hardware",
    "agency",
    "nonprofit",
    "government",
    "education",
)
SORTS = ("relevance", "discovered_asc", "discovered_desc")
FILTER_FACETS = (
    "sources",
    "industries",
    "skills",
    "company_categories",
    "job_categories",
    "job_subcategories",
    "occupations",
)

StrOrList = Union[str, List[str]]

_SANDBOX_NOTE = (
    "a sandbox API key (jbe_test_...) serves job search and job details only"
)


def _clean(values: List[Any]) -> List[str]:
    seen: List[str] = []
    for raw in values:
        value = str(raw).strip() if raw is not None else ""
        if value and value not in seen:
            seen.append(value)
    return seen


def _as_list(value: Optional[StrOrList], split: bool) -> List[str]:
    """A list argument. A single string is one value, or comma-split if `split`."""
    if value is None:
        return []
    if isinstance(value, str):
        return _clean(value.split(",") if split else [value])
    return _clean(list(value))


def _enum_list(
    name: str, value: Optional[StrOrList], allowed: Tuple[str, ...]
) -> List[str]:
    values = [v.lower() for v in _as_list(value, split=True)]
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown {name} value(s) {unknown}. Allowed: {', '.join(allowed)}. "
            "An unknown value would match no jobs."
        )
    return values


def _timestamp(name: str, value: Optional[str]) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an ISO 8601 timestamp such as 2026-01-31 or "
            f"2026-01-31T09:00:00Z, got {value!r}."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z")


def _pair(include: List[str], exclude: List[str]) -> Dict[str, List[str]]:
    pair: Dict[str, List[str]] = {}
    if include:
        pair["include"] = include
    if exclude:
        pair["exclude"] = exclude
    return pair


def _message(text: str, kind: str, **extra: Any) -> List[Document]:
    return [Document(text=text, metadata={"type": kind, **extra})]


class JoboToolSpec(BaseToolSpec):
    """
    Jobo tool spec: live job listings from employer career sites and ATS platforms.

    Search is billed per job returned. Looking up one job, companies, filter
    values and locations is free, though every call counts toward rate limits.

    Hiring signals (`get_hiring_signals`) are billed per signal, at ten times
    the price of a job, so that tool is only in the tool list when the spec is
    built with `include_hiring_signals=True`.
    """

    spec_functions = [
        ("search_jobs", "asearch_jobs"),
        ("get_job", "aget_job"),
        ("find_companies", "afind_companies"),
        ("find_filter_values", "afind_filter_values"),
        ("find_locations", "afind_locations"),
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_page_size: int = DEFAULT_MAX_PAGE_SIZE,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: Optional[httpx.Client] = None,
        http_async_client: Optional[httpx.AsyncClient] = None,
        include_hiring_signals: bool = False,
        max_signals: int = DEFAULT_MAX_SIGNALS,
    ) -> None:
        """
        Initialize with parameters.

        Args:
            api_key: Jobo API key. Read from the `JOBO_API_KEY` environment
                variable when not passed. It is only ever sent in the
                `X-Api-Key` header.
            max_page_size: Hard cap on jobs requested per `search_jobs` call
                (1-100, default 25). A larger `page_size` from the model is
                reduced to it, because every returned job is billed.
            base_url: API base URL. Defaults to `https://connect.jobo.world`.
            timeout: Response timeout in seconds.
            max_retries: Retries for 429, 500 and 503 responses and failures
                to connect. 402 and read timeouts are never retried, nor is a
                503 without `Retry-After`. Search retries only 429 and a 503
                with `Retry-After`: a 500 can come after search has billed.
            http_client: Optional `httpx.Client` for sync requests. It is
                never closed by the tool spec.
            http_async_client: Optional `httpx.AsyncClient` for async
                requests. It is never closed by the tool spec.
            include_hiring_signals: Add `get_hiring_signals` to this spec's
                tools. Off by default: every signal it returns is billed at
                ten times the price of a job, and a signal read again is
                billed again. Hiring signals need a live account key.
            max_signals: Hard cap on signals requested per
                `get_hiring_signals` call (1-200, default 25). A larger
                `limit` from the model is reduced to it.

        """
        if not 1 <= max_page_size <= API_MAX_PAGE_SIZE:
            raise ValueError(
                f"max_page_size must be between 1 and {API_MAX_PAGE_SIZE}."
            )
        if not 1 <= max_signals <= SIGNAL_MAX_LIMIT:
            raise ValueError(f"max_signals must be between 1 and {SIGNAL_MAX_LIMIT}.")
        self.max_page_size = max_page_size
        self.max_signals = max_signals
        if include_hiring_signals:
            # This instance only: the class keeps the job tools and free lookups.
            self.spec_functions = [*type(self).spec_functions, SIGNALS_SPEC_FUNCTION]
        self._client = JoboClient(
            api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            http_async_client=http_async_client,
        )
        self._taxonomy: Optional[Tuple[float, List[Dict[str, str]]]] = None

    def __repr__(self) -> str:
        return f"JoboToolSpec(max_page_size={self.max_page_size})"

    def close(self) -> None:
        """
        Close the HTTP connections this tool spec opened.

        The tool spec keeps one connection pool and reuses it for every call.
        An `http_client` you passed in is left open. The tool spec stays
        usable: its next call opens a new pool.
        """
        self._client.close()

    async def aclose(self) -> None:
        """
        Close the HTTP connections this tool spec opened, sync and async.

        Clients you passed in are left open. Await it on the event loop the
        tools ran on: each event loop gets a pool of its own.
        """
        await self._client.aclose()

    # ------------------------------------------------------------ search_jobs

    def _search_request(self, **a: Any) -> Tuple[Dict[str, Any], Optional[int]]:
        body: Dict[str, Any] = {}
        query = (a["query"] or "").strip()
        if query:
            body["queries"] = [query]
        locations = _as_list(a["location"], split=False)
        if locations:
            body["locations"] = locations
        if _as_list(a["sources"], split=True):
            body["sources"] = [s.lower() for s in _as_list(a["sources"], split=True)]
        for key, wire, values in (
            (
                "work_model",
                "work_models",
                _enum_list("work_model", a["work_model"], WORK_MODELS),
            ),
            (
                "employment_type",
                "employment_types",
                _enum_list("employment_type", a["employment_type"], EMPLOYMENT_TYPES),
            ),
            (
                "experience_level",
                "experience_levels",
                _enum_list(
                    "experience_level", a["experience_level"], EXPERIENCE_LEVELS
                ),
            ),
            (
                "language",
                "languages",
                [v.lower() for v in _as_list(a["language"], True)],
            ),
        ):
            if values:
                body[wire] = values
        pairs = {
            "skills": _pair(
                _as_list(a["skills"], True), _as_list(a["exclude_skills"], True)
            ),
            "companies": _pair(
                _as_list(a["companies"], False), _as_list(a["exclude_companies"], False)
            ),
            "industries": _pair(
                _as_list(a["industries"], False),
                _as_list(a["exclude_industries"], False),
            ),
            "company_categories": _pair(
                _enum_list(
                    "company_categories", a["company_categories"], COMPANY_CATEGORIES
                ),
                _enum_list(
                    "exclude_company_categories",
                    a["exclude_company_categories"],
                    COMPANY_CATEGORIES,
                ),
            ),
            "job_categories": _pair(
                _as_list(a["job_categories"], False),
                _as_list(a["exclude_job_categories"], False),
            ),
            "job_subcategories": _pair(
                _as_list(a["job_subcategories"], False),
                _as_list(a["exclude_job_subcategories"], False),
            ),
            "occupation_ids": _pair(
                _as_list(a["occupation_ids"], True),
                _as_list(a["exclude_occupation_ids"], True),
            ),
        }
        body.update({k: v for k, v in pairs.items() if v})
        salary: Dict[str, int] = {}
        for key, bound in (("min_salary_usd", "min"), ("max_salary_usd", "max")):
            if a[key] is not None and int(a[key]) < 0:
                raise ValueError(
                    f"{key} must be 0 or greater (0 means no bound), got {a[key]}."
                )
            if a[key]:
                salary[bound] = int(a[key])
        if salary:
            body["salary_usd"] = salary
        for key in (
            "posted_after",
            "posted_before",
            "discovered_after",
            "discovered_before",
        ):
            stamp = _timestamp(key, a[key])
            if stamp:
                body[key] = stamp
        if a["search_description"] is False:
            body["search_description"] = False
        sort = (a["sort"] or "").strip().lower()
        if sort:
            if sort not in SORTS:
                raise ValueError(f"Unknown sort {sort!r}. Allowed: {', '.join(SORTS)}.")
            if sort != "relevance":
                body["sort"] = sort
        page = max(int(a["page"]), 1)
        requested = max(int(a["page_size"]), 1)
        page_size = min(requested, self.max_page_size)
        body["page"] = page
        body["page_size"] = page_size
        return body, (page_size if page_size < requested else None)

    @staticmethod
    def _search_documents(reply: Reply, capped: Optional[int]) -> List[Document]:
        payload = reply.data if isinstance(reply.data, dict) else {}
        jobs = [j for j in (payload.get("jobs") or []) if isinstance(j, dict)]
        notes = search_notes(payload)
        if capped is not None:
            notes.append(
                f"Note: page_size was capped at {capped}, this tool's configured maximum."
            )
        summary = "\n".join(
            [search_header(payload, len(jobs)) if jobs else EMPTY_SEARCH, *notes]
        )
        meta: Dict[str, Any] = {
            "type": "search_summary",
            "total": payload.get("total"),
            "page": payload.get("page"),
            "page_size": payload.get("page_size"),
            "total_pages": payload.get("total_pages"),
            **reply.usage,
        }
        documents = [
            Document(
                text=summary, metadata={k: v for k, v in meta.items() if v is not None}
            )
        ]
        for index, job in enumerate(jobs, start=1):
            documents.append(
                Document(
                    doc_id=str(job.get("id") or "") or None,
                    text=format_job_summary(job, index),
                    metadata={"type": "job", **job_metadata(job)},
                )
            )
        return documents

    def search_jobs(
        self,
        query: Optional[str] = None,
        location: Optional[StrOrList] = None,
        work_model: Optional[StrOrList] = None,
        employment_type: Optional[StrOrList] = None,
        experience_level: Optional[StrOrList] = None,
        skills: Optional[StrOrList] = None,
        exclude_skills: Optional[StrOrList] = None,
        companies: Optional[StrOrList] = None,
        exclude_companies: Optional[StrOrList] = None,
        industries: Optional[StrOrList] = None,
        exclude_industries: Optional[StrOrList] = None,
        job_categories: Optional[StrOrList] = None,
        exclude_job_categories: Optional[StrOrList] = None,
        job_subcategories: Optional[StrOrList] = None,
        exclude_job_subcategories: Optional[StrOrList] = None,
        occupation_ids: Optional[StrOrList] = None,
        exclude_occupation_ids: Optional[StrOrList] = None,
        sources: Optional[StrOrList] = None,
        language: Optional[StrOrList] = None,
        company_categories: Optional[StrOrList] = None,
        exclude_company_categories: Optional[StrOrList] = None,
        min_salary_usd: Optional[int] = None,
        max_salary_usd: Optional[int] = None,
        posted_after: Optional[str] = None,
        posted_before: Optional[str] = None,
        discovered_after: Optional[str] = None,
        discovered_before: Optional[str] = None,
        search_description: Optional[bool] = None,
        sort: Optional[str] = None,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> List[Document]:
        """
        Search live job listings. BILLED per job returned, so keep page_size small.

        Searches jobs from employer career sites and ATS platforms. Every job
        returned is billed (an empty result costs nothing), so do not repeat a
        search you already have the answer to. Jobs included in the account's
        plan are used before the wallet is charged. Filters combine with AND;
        several values inside one filter combine with OR. A filter value the
        index does not hold silently matches nothing, so look values up first
        (free) with find_filter_values, find_companies and find_locations.

        Returns a summary (total matches, page, warnings) and one entry per job
        with ID, title, company, location, salary when stated, posted date and
        URL. Use get_job with an ID for the full posting.

        Args:
            query: Free-text query, e.g. a job title or skill. Wrap it in
                double quotes for an exact title phrase match.
            location: A city, region or country such as "Berlin, Germany", or
                a list of places (a job in any of them matches).
            work_model: Any of remote, hybrid, onsite.
            employment_type: Any of full-time, part-time, contract,
                internship, freelance, temporary.
            experience_level: Any of intern, entry, mid, senior, lead,
                executive.
            skills: Require skills such as Python or Kubernetes.
            exclude_skills: Leave out jobs asking for any of these skills.
            companies: Only jobs at these companies. Each value is a Jobo
                company id, a website domain ("stripe.com") or a whole
                company name. Names match whole, never partially: use the
                ids from find_companies for an exact match.
            exclude_companies: Leave out jobs at these companies (ids,
                domains or whole names).
            industries: Industry names, exactly as find_filter_values lists
                them (facet industries).
            exclude_industries: Leave out companies in these industries.
            job_categories: Occupation category names such as "Healthcare"
                (facet job_categories).
            exclude_job_categories: Leave out these occupation categories.
            job_subcategories: Occupation subcategory names such as "Nursing"
                (facet job_subcategories).
            exclude_job_subcategories: Leave out these subcategories.
            occupation_ids: Occupation ids, not names (facet occupations
                with the occupation's name as the query).
            exclude_occupation_ids: Leave out these occupation ids.
            sources: ATS provider ids such as greenhouse, lever, workday
                (facet sources).
            language: Description languages as ISO 639-1 codes, e.g. en, de.
            company_categories: Hiring company's business model: b2b, b2c,
                b2b2c, b2g, saas, marketplace, e-commerce, service-provider,
                hardware, agency, nonprofit, government, education.
            exclude_company_categories: Leave out these business models,
                e.g. agency.
            min_salary_usd: Minimum annual salary in USD, 0 or more. Jobs
                without disclosed pay are excluded when a salary bound is set.
            max_salary_usd: Maximum annual salary in USD, 0 or more.
            posted_after: Only jobs posted on or after this ISO 8601 date or
                timestamp.
            posted_before: Only jobs posted on or before this ISO 8601 date or
                timestamp.
            discovered_after: Only jobs Jobo first indexed on or after this
                ISO 8601 date or timestamp.
            discovered_before: Only jobs Jobo first indexed on or before this
                ISO 8601 date or timestamp.
            search_description: false matches the query against job titles
                only; true (default) also against company, skills and summary.
            sort: relevance (default), discovered_asc or discovered_desc
                (when Jobo first indexed the job).
            page: Page number, starting at 1.
            page_size: Jobs to return (default 5). Every returned job is
                billed. Reduced to this tool's configured maximum if larger.

        """
        body, capped = self._search_request(**self._args(locals()))
        reply = self._client.request("POST", "/api/jobs/search", json=body)
        return self._search_documents(reply, capped)

    async def asearch_jobs(
        self,
        query: Optional[str] = None,
        location: Optional[StrOrList] = None,
        work_model: Optional[StrOrList] = None,
        employment_type: Optional[StrOrList] = None,
        experience_level: Optional[StrOrList] = None,
        skills: Optional[StrOrList] = None,
        exclude_skills: Optional[StrOrList] = None,
        companies: Optional[StrOrList] = None,
        exclude_companies: Optional[StrOrList] = None,
        industries: Optional[StrOrList] = None,
        exclude_industries: Optional[StrOrList] = None,
        job_categories: Optional[StrOrList] = None,
        exclude_job_categories: Optional[StrOrList] = None,
        job_subcategories: Optional[StrOrList] = None,
        exclude_job_subcategories: Optional[StrOrList] = None,
        occupation_ids: Optional[StrOrList] = None,
        exclude_occupation_ids: Optional[StrOrList] = None,
        sources: Optional[StrOrList] = None,
        language: Optional[StrOrList] = None,
        company_categories: Optional[StrOrList] = None,
        exclude_company_categories: Optional[StrOrList] = None,
        min_salary_usd: Optional[int] = None,
        max_salary_usd: Optional[int] = None,
        posted_after: Optional[str] = None,
        posted_before: Optional[str] = None,
        discovered_after: Optional[str] = None,
        discovered_before: Optional[str] = None,
        search_description: Optional[bool] = None,
        sort: Optional[str] = None,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> List[Document]:
        """Async version of `search_jobs`. Billed per job returned, like the sync one."""
        body, capped = self._search_request(**self._args(locals()))
        reply = await self._client.arequest("POST", "/api/jobs/search", json=body)
        return self._search_documents(reply, capped)

    @staticmethod
    def _args(local_vars: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in local_vars.items() if k != "self"}

    # -------------------------------------------------------------- get_job

    @staticmethod
    def _job_path(job_id: str) -> Optional[str]:
        try:
            return f"/api/jobs/{quote(str(UUID(job_id.strip())), safe='')}"
        except ValueError:
            return None

    @staticmethod
    def _not_an_id(job_id: str) -> List[Document]:
        return _message(
            f'"{job_id.strip()}" is not a Jobo job id. Job ids are UUIDs: the `ID` '
            "values search_jobs returns.",
            "job_error",
        )

    @staticmethod
    def _job_documents(reply: Reply, include_description: bool) -> List[Document]:
        job = reply.data if isinstance(reply.data, dict) else {}
        return [
            Document(
                doc_id=str(job.get("id") or "") or None,
                text=format_job_document(job, include_description=include_description),
                metadata={"type": "job", **job_metadata(job)},
            )
        ]

    @staticmethod
    def _recover_job(job_id: str, error: JoboAPIError) -> List[Document]:
        if isinstance(error, JoboNotFoundError) and _is_job_not_found(error):
            return _message(
                f'No job exists with id "{job_id.strip()}". It may have closed. '
                "Job ids are the `ID` values search_jobs returns.",
                "job_error",
            )
        raise error

    def get_job(self, job_id: str, include_description: bool = True) -> List[Document]:
        """
        Get the full details of one job listing by its Jobo job ID. Free: not billed.
        Returns the company, location, compensation, responsibilities,
        qualifications, benefits, the employer's description and the listing
        and apply URLs. Use an ID returned by search_jobs. An ID that is
        unknown or has closed comes back as a plain message, not an error.

        Args:
            job_id: The Jobo job id (a UUID), as returned by search_jobs.
            include_description: Append the employer's full description
                after the structured fields.

        """
        path = self._job_path(job_id)
        if path is None:
            return self._not_an_id(job_id)
        try:
            reply = self._client.request("GET", path)
        except JoboAPIError as error:
            return self._recover_job(job_id, error)
        return self._job_documents(reply, include_description)

    async def aget_job(
        self, job_id: str, include_description: bool = True
    ) -> List[Document]:
        """Async version of `get_job`. Free: not billed."""
        path = self._job_path(job_id)
        if path is None:
            return self._not_an_id(job_id)
        try:
            reply = await self._client.arequest("GET", path)
        except JoboAPIError as error:
            return self._recover_job(job_id, error)
        return self._job_documents(reply, include_description)

    # -------------------------------------------------------- find_companies

    def _sandbox_unavailable(
        self, error: JoboAPIError, what: str, advice: str
    ) -> List[Document]:
        if isinstance(error, JoboNotFoundError) and error.not_available_in_sandbox:
            return _message(
                f"{what} cannot be looked up here: {_SANDBOX_NOTE}. Do not call "
                f"this tool again. {advice}",
                "lookup_unavailable",
            )
        raise error

    @staticmethod
    def _items(reply: Reply, key: str) -> List[Dict[str, Any]]:
        payload = reply.data if isinstance(reply.data, dict) else {}
        return [i for i in (payload.get(key) or []) if isinstance(i, dict)]

    def _company_documents(self, query: str, reply: Reply) -> List[Document]:
        return _message(
            format_company_suggestions(query, self._items(reply, "suggestions")),
            "company_lookup",
        )

    def find_companies(self, query: str, limit: int = 5) -> List[Document]:
        """
        Look up companies by the start of their name or website domain. Free: not billed.
        Returns each matching company's name, domain, country, live job count
        and Jobo company ids. Pass all of a company's ids as the `companies`
        filter of search_jobs to search that company's jobs exactly (a name
        alone matches every company with that name).

        Args:
            query: The start of a company name or of its website domain, at
                least 2 characters, e.g. "strip" or "stripe.com".
            limit: Maximum number of companies to return (1-20).

        """
        params: Dict[str, Any] = {"q": query.strip(), "limit": min(max(limit, 1), 20)}
        try:
            reply = self._client.request(
                "GET", "/api/connectors/companies/suggest", params=params
            )
        except JoboAPIError as error:
            return self._sandbox_unavailable(
                error,
                "Companies",
                "Pass the company's name or website domain as the `companies` "
                "filter of search_jobs instead.",
            )
        return self._company_documents(str(params["q"]), reply)

    async def afind_companies(self, query: str, limit: int = 5) -> List[Document]:
        """Async version of `find_companies`. Free: not billed."""
        params: Dict[str, Any] = {"q": query.strip(), "limit": min(max(limit, 1), 20)}
        try:
            reply = await self._client.arequest(
                "GET", "/api/connectors/companies/suggest", params=params
            )
        except JoboAPIError as error:
            return self._sandbox_unavailable(
                error,
                "Companies",
                "Pass the company's name or website domain as the `companies` "
                "filter of search_jobs instead.",
            )
        return self._company_documents(str(params["q"]), reply)

    # ---------------------------------------------------- find_filter_values

    @staticmethod
    def _facet(facet: str) -> str:
        name = facet.strip().lower()
        if name not in FILTER_FACETS:
            raise ValueError(
                f"Unknown facet {facet!r}. Allowed: {', '.join(FILTER_FACETS)}."
            )
        return name

    def _cached_taxonomy(self) -> Optional[List[Dict[str, str]]]:
        if self._taxonomy is None:
            return None
        loaded_at, rows = self._taxonomy
        if time.monotonic() - loaded_at > _TAXONOMY_TTL_SECONDS:
            return None
        return rows

    def _keep_taxonomy(self, reply: Reply) -> List[Dict[str, str]]:
        rows = flatten_taxonomy(reply.data)
        self._taxonomy = (time.monotonic(), rows)
        return rows

    @staticmethod
    def _occupation_documents(
        query: str, limit: int, rows: List[Dict[str, str]]
    ) -> List[Document]:
        return _message(
            format_occupation_matches(query, match_occupations(rows, query, limit)),
            "filter_values",
            facet="occupations",
        )

    @staticmethod
    def _options_request(
        facet: str, query: str, limit: int
    ) -> Tuple[str, Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": min(max(limit, 1), 50)}
        if query:
            params["q"] = query
        return f"/api/connectors/filter-options/{quote(facet, safe='')}", params

    def _option_documents(self, facet: str, query: str, reply: Reply) -> List[Document]:
        return _message(
            format_filter_values(facet, query, self._items(reply, "options")),
            "filter_values",
            facet=facet,
        )

    def _filter_unavailable(self, error: JoboAPIError) -> List[Document]:
        return self._sandbox_unavailable(
            error,
            "Filter values",
            "Call search_jobs without the lookup; work_model, employment_type, "
            "experience_level and company_categories list their values in that "
            "tool's arguments.",
        )

    def find_filter_values(
        self, facet: str, query: Optional[str] = None, limit: int = 20
    ) -> List[Document]:
        """
        List the values a search_jobs filter accepts, with live job counts. Free: not billed.
        Call it before filtering on one of these, because a value the index
        does not hold matches no jobs and the API does not say so. For
        occupations, pass the occupation's name as the query to get the ids
        the `occupation_ids` filter takes. For companies use find_companies.

        Args:
            facet: Which filter's values to list: sources (ATS providers),
                industries, skills, company_categories, job_categories,
                job_subcategories or occupations.
            query: Optional text to narrow the list. For occupations it is
                searched in the occupation names, e.g. "nurse". For the other
                facets a value must start with it, e.g. "soft". Leave it out
                for the most common values.
            limit: Maximum number of values to return (1-50).

        """
        name = self._facet(facet)
        text = (query or "").strip()
        try:
            if name == "occupations" and text:
                rows = self._cached_taxonomy()
                if rows is None:
                    rows = self._keep_taxonomy(
                        self._client.request("GET", "/api/occupations")
                    )
                return self._occupation_documents(text, min(max(limit, 1), 50), rows)
            path, params = self._options_request(name, text, limit)
            reply = self._client.request("GET", path, params=params)
        except JoboAPIError as error:
            return self._filter_unavailable(error)
        return self._option_documents(name, text, reply)

    async def afind_filter_values(
        self, facet: str, query: Optional[str] = None, limit: int = 20
    ) -> List[Document]:
        """Async version of `find_filter_values`. Free: not billed."""
        name = self._facet(facet)
        text = (query or "").strip()
        try:
            if name == "occupations" and text:
                rows = self._cached_taxonomy()
                if rows is None:
                    rows = self._keep_taxonomy(
                        await self._client.arequest("GET", "/api/occupations")
                    )
                return self._occupation_documents(text, min(max(limit, 1), 50), rows)
            path, params = self._options_request(name, text, limit)
            reply = await self._client.arequest("GET", path, params=params)
        except JoboAPIError as error:
            return self._filter_unavailable(error)
        return self._option_documents(name, text, reply)

    # --------------------------------------------------------- find_locations

    def _location_documents(self, query: str, reply: Reply) -> List[Document]:
        return _message(
            format_location_suggestions(query, self._items(reply, "suggestions")),
            "location_lookup",
        )

    def _location_unavailable(self, error: JoboAPIError) -> List[Document]:
        return self._sandbox_unavailable(
            error,
            "Places",
            "Pass the place name as the `location` filter of search_jobs as it "
            "is: the search resolves free-form locations itself.",
        )

    def find_locations(self, query: str, limit: int = 5) -> List[Document]:
        """
        Resolve a place name to the locations search_jobs takes. Free: not billed.
        Returns matching cities, regions and countries, written the way the
        `location` filter of search_jobs takes them. search_jobs already
        resolves a free-form location by itself, so use this for an ambiguous
        name such as "Springfield", or when a search reports a location it
        could not resolve.

        Args:
            query: A place name or the start of one, at least 2 characters,
                e.g. "Springfield" or "Berl".
            limit: Maximum number of places to return (1-10).

        """
        params: Dict[str, Any] = {"q": query.strip(), "limit": min(max(limit, 1), 10)}
        try:
            reply = self._client.request(
                "GET", "/api/connectors/locations/suggest", params=params
            )
        except JoboAPIError as error:
            return self._location_unavailable(error)
        return self._location_documents(str(params["q"]), reply)

    async def afind_locations(self, query: str, limit: int = 5) -> List[Document]:
        """Async version of `find_locations`. Free: not billed."""
        params: Dict[str, Any] = {"q": query.strip(), "limit": min(max(limit, 1), 10)}
        try:
            reply = await self._client.arequest(
                "GET", "/api/connectors/locations/suggest", params=params
            )
        except JoboAPIError as error:
            return self._location_unavailable(error)
        return self._location_documents(str(params["q"]), reply)

    # ----------------------------------------------------- get_hiring_signals

    def _signals_request(self, **a: Any) -> Tuple[str, Dict[str, Any], Optional[int]]:
        filters: Dict[str, Any] = {}
        for key, allowed in (
            ("types", SIGNAL_TYPES),
            ("company_sizes", COMPANY_SIZES),
            ("seniority", SIGNAL_SENIORITY),
        ):
            filters[key] = check_enum(key, _as_list(a[key], True), allowed)
        for key in ("domains", "countries", "technologies", "skills", "job_countries"):
            filters[key] = _as_list(a[key], True)
        # A name is one value: a comma in it is refused, never split.
        for key in ("industries", "funding_stages", "functions", "roles"):
            filters[key] = _as_list(a[key], False)
        for key in ("detected_after", "detected_before"):
            filters[key] = check_day(key, a[key])
        filters["created_after"] = check_instant("created_after", a["created_after"])
        strength = a["min_strength"]
        if strength is not None and not 1 <= int(strength) <= 100:
            raise ValueError(f"min_strength must be 1-100, got {strength}.")
        filters["min_strength"] = None if strength is None else int(strength)
        filters["exclude_agencies"] = a["exclude_agencies"] is True
        requested = max(int(a["limit"]), 1)
        limit = min(requested, self.max_signals)
        path, params = build_signals_request(
            filters, company_id=a["company_id"], limit=limit, cursor=a["cursor"]
        )
        return path, params, (limit if limit < requested else None)

    @staticmethod
    def _signals_documents(reply: Reply, capped: Optional[int]) -> List[Document]:
        payload = reply.data if isinstance(reply.data, dict) else {}
        signals = signal_list(payload)
        cursor = next_cursor(payload)
        lines = [signals_header(len(signals), cursor)]
        if capped is not None:
            lines.append(
                f"Note: limit was capped at {capped}, this tool's configured maximum."
            )
        summary: Dict[str, Any] = {
            "type": "signals_summary",
            "signals": len(signals),
            "has_more": cursor is not None,
            "next_cursor": cursor,
            **reply.usage,
        }
        documents = [
            Document(
                text="\n".join(lines),
                metadata={k: v for k, v in summary.items() if v is not None},
            )
        ]
        for index, signal in enumerate(signals, start=1):
            documents.append(
                Document(
                    doc_id=str(signal.get("id") or "") or None,
                    text=format_signal(signal, index),
                    metadata={"type": "hiring_signal", **signal_metadata(signal)},
                )
            )
        documents.append(
            Document(
                text=signals_footer(signals, reply.usage, "get_job"),
                metadata={
                    "type": "signals_billing",
                    "billed_signals": len(signals),
                    **reply.usage,
                },
            )
        )
        return documents

    @staticmethod
    def _signals_failure(error: JoboAPIError) -> List[Document]:
        if isinstance(error, JoboNotFoundError) and error.not_available_in_sandbox:
            return _message(NO_SIGNALS_IN_SANDBOX, "signals_unavailable")
        advice = signals_error_advice(error)
        if advice:
            error.args = (f"{error}\n{advice}",)
        raise error

    def get_hiring_signals(
        self,
        types: Optional[StrOrList] = None,
        domains: Optional[StrOrList] = None,
        company_id: Optional[str] = None,
        countries: Optional[StrOrList] = None,
        company_sizes: Optional[StrOrList] = None,
        industries: Optional[StrOrList] = None,
        funding_stages: Optional[StrOrList] = None,
        functions: Optional[StrOrList] = None,
        roles: Optional[StrOrList] = None,
        seniority: Optional[StrOrList] = None,
        technologies: Optional[StrOrList] = None,
        skills: Optional[StrOrList] = None,
        job_countries: Optional[StrOrList] = None,
        min_strength: Optional[int] = None,
        detected_after: Optional[str] = None,
        detected_before: Optional[str] = None,
        created_after: Optional[str] = None,
        exclude_agencies: Optional[bool] = None,
        limit: int = DEFAULT_SIGNAL_LIMIT,
        cursor: Optional[str] = None,
    ) -> List[Document]:
        """
        Read hiring signals. BILLED per signal returned, at ten times a job's price.

        A hiring signal is one event at one company, detected from its job
        postings: a hiring surge, a first job in a new country or function, a
        leadership search, a new technology, a slowdown or a hiring freeze. Use
        it to find companies that are growing, entering a market or building a
        team, or to check whether one company is hiring or freezing.

        Every signal returned is billed at ten times the price of a job, and a
        signal read again is billed again: keep limit small and do not repeat
        a call you already have the answer to. A call that returns no signals
        costs nothing. Filters combine with AND; several values inside one
        filter combine with OR. A value cannot contain a comma.

        Returns a summary (how many signals, and the cursor of the next page),
        one entry per signal with its company and domain, type, a one-line
        summary, the day it was detected, its strength and the ids of the jobs
        behind it (read them free with get_job), and a last line saying what
        was billed.

        Args:
            types: Only these kinds of signal: hiring_surge, new_function,
                new_country, leadership_hire, new_skill, hiring_slowdown,
                first_role, new_technology, function_surge, new_city,
                hard_to_fill, leadership_role_closed, hiring_freeze,
                hiring_resumed. Leave empty for every kind.
            domains: Only signals at these companies, by website domain, e.g.
                "stripe.com". At most 100.
            company_id: One company's signals, by Jobo company id (a UUID): the
                Company ID shown with each signal. It takes only types,
                min_strength, detected_after, detected_before, created_after,
                limit and cursor. To name companies by website, use domains.
            countries: Countries the companies are headquartered in, as
                two-letter codes, e.g. "US", "DE".
            company_sizes: Headcount bands of the companies: 1-10, 11-50,
                51-200, 201-500, 501-1000, 1001-5000, 5001-10000, 10001+.
            industries: Industries of the companies, matched against the
                primary industry and the industry list.
            funding_stages: Latest funding round of the companies, e.g. "Seed",
                "Series A".
            functions: Job categories the signal must be about, e.g.
                "Technology", "Sales". A focus filter.
            roles: Words or phrases to look for in the job title or occupation
                behind the signal, e.g. "data engineer". At least 2 characters
                each. A focus filter.
            seniority: Seniority of the role behind the signal: junior, mid,
                senior, director, vp, c_level, executive. A focus filter.
            technologies: Named tools the signal must be about, e.g.
                "Snowflake". A focus filter, combined with skills.
            skills: Named skills the signal must be about. A focus filter,
                combined with technologies.
            job_countries: Countries the jobs are in (not the company's
                headquarters), as two-letter codes. A focus filter.
            min_strength: Only signals at least this strong, 1-100. Strength
                compares signals of the same type.
            detected_after: Earliest detection day, inclusive, as yyyy-MM-dd.
            detected_before: Latest detection day, inclusive, as yyyy-MM-dd.
            created_after: Only signals first stored after this instant,
                exclusive, e.g. 2026-11-12T09:30:00Z.
            exclude_agencies: Leave out staffing agencies, whose postings are
                their clients' hiring.
            limit: Signals to return (default 5). Every returned signal is
                billed. Reduced to this tool's configured maximum if larger.
            cursor: The cursor a previous answer gave, to read its next page.
                Pass the same other arguments with it.

        """
        path, params, capped = self._signals_request(**self._args(locals()))
        try:
            reply = self._client.request("GET", path, params=params)
        except JoboAPIError as error:
            return self._signals_failure(error)
        return self._signals_documents(reply, capped)

    async def aget_hiring_signals(
        self,
        types: Optional[StrOrList] = None,
        domains: Optional[StrOrList] = None,
        company_id: Optional[str] = None,
        countries: Optional[StrOrList] = None,
        company_sizes: Optional[StrOrList] = None,
        industries: Optional[StrOrList] = None,
        funding_stages: Optional[StrOrList] = None,
        functions: Optional[StrOrList] = None,
        roles: Optional[StrOrList] = None,
        seniority: Optional[StrOrList] = None,
        technologies: Optional[StrOrList] = None,
        skills: Optional[StrOrList] = None,
        job_countries: Optional[StrOrList] = None,
        min_strength: Optional[int] = None,
        detected_after: Optional[str] = None,
        detected_before: Optional[str] = None,
        created_after: Optional[str] = None,
        exclude_agencies: Optional[bool] = None,
        limit: int = DEFAULT_SIGNAL_LIMIT,
        cursor: Optional[str] = None,
    ) -> List[Document]:
        """Async version of `get_hiring_signals`. Billed per signal returned."""
        path, params, capped = self._signals_request(**self._args(locals()))
        try:
            reply = await self._client.arequest("GET", path, params=params)
        except JoboAPIError as error:
            return self._signals_failure(error)
        return self._signals_documents(reply, capped)


def _is_job_not_found(error: JoboNotFoundError) -> bool:
    """
    Whether the API itself said that no job has this id.

    `GET /api/jobs/{id}` answers an unknown id with the problem "Job not
    found". Every other 404 (a wrong base_url, a proxy, a sandbox gap) means
    something is misconfigured and must not be read as "no such job".
    """
    if error.code == "job_not_found":
        return True
    title = error.body.get("title") if isinstance(error.body, dict) else None
    return isinstance(title, str) and title.strip().lower() == "job not found"
