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
from llama_index.tools.jobo.errors import JoboAPIError, JoboNotFoundError

API_MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 5
DEFAULT_MAX_PAGE_SIZE = 25
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
                to connect. 402 and read timeouts are never retried.
            http_client: Optional `httpx.Client` for sync requests.
            http_async_client: Optional `httpx.AsyncClient` for async requests.

        """
        if not 1 <= max_page_size <= API_MAX_PAGE_SIZE:
            raise ValueError(
                f"max_page_size must be between 1 and {API_MAX_PAGE_SIZE}."
            )
        self.max_page_size = max_page_size
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
            "industries": _pair(_as_list(a["industries"], False), []),
            "company_categories": _pair(
                _enum_list(
                    "company_categories", a["company_categories"], COMPANY_CATEGORIES
                ),
                [],
            ),
            "job_categories": _pair(_as_list(a["job_categories"], False), []),
            "occupation_ids": _pair(_as_list(a["occupation_ids"], True), []),
        }
        body.update({k: v for k, v in pairs.items() if v})
        salary: Dict[str, int] = {}
        if a["min_salary_usd"]:
            salary["min"] = int(a["min_salary_usd"])
        if a["max_salary_usd"]:
            salary["max"] = int(a["max_salary_usd"])
        if salary:
            body["salary_usd"] = salary
        for key in ("posted_after", "posted_before"):
            stamp = _timestamp(key, a[key])
            if stamp:
                body[key] = stamp
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
        job_categories: Optional[StrOrList] = None,
        occupation_ids: Optional[StrOrList] = None,
        sources: Optional[StrOrList] = None,
        language: Optional[StrOrList] = None,
        company_categories: Optional[StrOrList] = None,
        min_salary_usd: Optional[int] = None,
        max_salary_usd: Optional[int] = None,
        posted_after: Optional[str] = None,
        posted_before: Optional[str] = None,
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
            job_categories: Occupation category names such as "Healthcare"
                (facet job_categories).
            occupation_ids: Occupation ids, not names (facet occupations
                with the occupation's name as the query).
            sources: ATS provider ids such as greenhouse, lever, workday
                (facet sources).
            language: Description languages as ISO 639-1 codes, e.g. en, de.
            company_categories: Hiring company's business model: b2b, b2c,
                b2b2c, b2g, saas, marketplace, e-commerce, service-provider,
                hardware, agency, nonprofit, government, education.
            min_salary_usd: Minimum annual salary in USD. Jobs without
                disclosed pay are excluded when a salary bound is set.
            max_salary_usd: Maximum annual salary in USD.
            posted_after: Only jobs posted on or after this ISO 8601 date or
                timestamp.
            posted_before: Only jobs posted on or before this ISO 8601 date or
                timestamp.
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
        job_categories: Optional[StrOrList] = None,
        occupation_ids: Optional[StrOrList] = None,
        sources: Optional[StrOrList] = None,
        language: Optional[StrOrList] = None,
        company_categories: Optional[StrOrList] = None,
        min_salary_usd: Optional[int] = None,
        max_salary_usd: Optional[int] = None,
        posted_after: Optional[str] = None,
        posted_before: Optional[str] = None,
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
