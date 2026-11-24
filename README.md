# LlamaIndex Tools Integration: Jobo

[Jobo](https://jobo.world) is a jobs-data API: live job listings collected directly from employer career sites and applicant tracking systems (ATS). This package gives LlamaIndex agents tools to search those listings, read one in full, and look up the companies, filter values and locations a search needs.

- Maintainer: Jobo (`support@jobo.world`)
- Homepage: <https://jobo.world>
- Docs: <https://jobo.world/docs>

## Installation

```bash
pip install llama-index-tools-jobo
```

Requires Python 3.11+. The package depends on `llama-index-core` and `httpx`.

## Authentication

Create an API key in the [Jobo dashboard](https://enterprise.jobo.world/api-keys) under **Developers > API Keys**, then either export it:

```bash
export JOBO_API_KEY="jbe_live_..."
```

or pass `api_key="..."` to `JoboToolSpec`. The key is only ever sent in the `X-Api-Key` request header: it is not stored on a public attribute, shown in `repr()`, or logged. A missing or blank key raises `ValueError` when the tool spec is built, not on the first request.

## Usage

```python
import asyncio

from llama_index.tools.jobo import JoboToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI


async def main() -> None:
    tool_spec = JoboToolSpec(max_page_size=10)
    agent = FunctionAgent(
        tools=tool_spec.to_tool_list(),
        llm=OpenAI(model="gpt-4.1"),
        system_prompt=(
            "You help people find jobs. Look up filter values and companies before "
            "you filter on them, search with search_jobs, read a posting with "
            "get_job before describing it, and cite each job's listing URL."
        ),
    )
    print(await agent.run("Find three remote senior Python jobs paying over $150k."))
    await tool_spec.aclose()


asyncio.run(main())
```

In a notebook, where an event loop is already running, `await main()` instead of `asyncio.run(main())`.

The tools can also be called directly:

```python
from llama_index.tools.jobo import JoboToolSpec

tool_spec = JoboToolSpec()

documents = tool_spec.search_jobs(
    query="backend engineer",
    location="Berlin, Germany",
    work_model=["remote", "hybrid"],
    experience_level=["senior"],
    exclude_skills=["PHP"],
    min_salary_usd=90000,
    page_size=3,
)
print(documents[0].text)  # summary: total matches, page, warnings
print(documents[1].text)  # the first job: id, title, company, URL, ...
```

## Available tool functions

| Function | What it does | Jobo endpoint | Billed |
| --- | --- | --- | --- |
| `search_jobs` | Search jobs with the full filter set | `POST /api/jobs/search` | Per job returned |
| `get_job` | One listing by id, as readable prose | `GET /api/jobs/{id}` | Free |
| `find_companies` | Company name or domain to company ids | `GET /api/connectors/companies/suggest` | Free |
| `find_filter_values` | Values a filter accepts (sources, industries, skills, company categories, job categories, job subcategories, occupations), with live job counts | `GET /api/connectors/filter-options/{facet}`, `GET /api/occupations` | Free |
| `find_locations` | A place name to the locations `search_jobs` takes | `GET /api/connectors/locations/suggest` | Free |

Each function also has an async twin (`asearch_jobs`, `aget_job`, `afind_companies`, `afind_filter_values`, `afind_locations`), which `to_tool_list()` registers as the tool's async function.

Every function returns `List[Document]`. `search_jobs` returns a first document with `metadata["type"] == "search_summary"` (total matches, page, warnings, and the credits the API reported for the call), then one `type == "job"` document per job whose `doc_id` is the Jobo job id. `get_job` returns one job document. The lookups return one document of text. Called as a tool, each document reaches the model as its text: `llama-index-core` 0.13.1 and later turn a returned `Document` into its content, which is why that is the lowest version this package accepts.

### `search_jobs` filters

`query`, `location`, `work_model`, `employment_type`, `experience_level`, `skills`, `exclude_skills`, `companies`, `exclude_companies`, `industries`, `exclude_industries`, `job_categories`, `exclude_job_categories`, `job_subcategories`, `exclude_job_subcategories`, `occupation_ids`, `exclude_occupation_ids`, `sources`, `language`, `company_categories`, `exclude_company_categories`, `min_salary_usd`, `max_salary_usd`, `posted_after`, `posted_before`, `discovered_after`, `discovered_before`, `search_description`, `sort`, `page` and `page_size`. Pass them by name: their order is not part of the API.

- Filters combine with AND; the values inside one list filter combine with OR.
- **Fixed-value filters** take lowercase keys: `work_model` is `remote`, `hybrid` or `onsite`; `employment_type` is `full-time`, `part-time`, `contract`, `internship`, `freelance` or `temporary`; `experience_level` is `intern`, `entry`, `mid`, `senior`, `lead` or `executive`; `company_categories` and `exclude_company_categories` are `b2b`, `b2c`, `b2b2c`, `b2g`, `saas`, `marketplace`, `e-commerce`, `service-provider`, `hardware`, `agency`, `nonprofit`, `government` or `education`. An unknown value raises `ValueError` before any request, because the API would answer it with zero results rather than an error.
- **Open-value filters** (`sources`, `skills`, `industries`, `job_categories`, `job_subcategories`, `occupation_ids`, `companies`, and their `exclude_` twins) match only values the index holds. Use `find_filter_values` and `find_companies` (both free) to get them.
- A list argument also accepts a single string. For ids, codes, fixed values and skills it is split on commas. For names (`companies`, `industries`, `job_categories`, `job_subcategories` and their `exclude_` twins) it is one value: `"FedEx, Inc."` is one company.
- `min_salary_usd` and `max_salary_usd` are 0 or more (0 means no bound); a negative value raises `ValueError` before any request.
- `discovered_after` and `discovered_before` bound when Jobo first indexed the job; `posted_after` and `posted_before` bound the employer's posting date.
- `search_description=False` matches `query` against job titles only. By default it also matches the company, skills and summary.
- `description_format` from the API is not an argument: search results carry no description. `get_job` returns the full posting.
- `page_size` defaults to 5 and never exceeds `max_page_size` (default 25, at most 100, the API's own limit). A larger request is reduced to the cap, with a note in the summary document.

## Billing

Jobo bills per job, not per request. These notes follow the [billing docs](https://jobo.world/docs/billing), which are the source of truth.

- **Search bills every job a response returns**, including a job an earlier search already returned. An empty result costs nothing.
- A Job Search plan's included jobs are used first, then the prepaid wallet at the pay-as-you-go rate of $3.00 per 1,000 jobs. The Unlimited plan has no per-job charge on Search, within the rate limits.
- Before a billed request runs, the API checks that included jobs plus the wallet cover a full page (`page_size`) and answers `402` if they do not. A smaller `page_size` needs less cover.
- `get_job`, `find_companies`, `find_filter_values` and `find_locations` are free. They still count toward rate limits.
- `find_locations` uses the free connector typeahead. It is not the metered `GET /api/locations/geocode` endpoint, which this package does not call.

What this package does about it: the agent asks for 5 jobs by default, every call is capped at `max_page_size`, a `402` is never retried, and neither is a read timeout or a `500` on search (a billed page may already have been delivered). An agent that searches in a loop pays for every job each call returns, so keep `max_page_size` low and tell the agent when to stop searching.

### Sandbox keys

A sandbox key (`jbe_test_...`) is free and returns fictional jobs in the real response shape. The sandbox serves job search and job lookups only. With a sandbox key, `find_companies`, `find_filter_values` and `find_locations` do not raise: they return a message telling the agent the lookup is unavailable and what to do instead. See the [sandbox docs](https://jobo.world/docs/sandbox).

## Errors and retries

Requests that fail with `429`, `500` or `503`, or that cannot connect, are retried up to `max_retries` times (default 2). `Retry-After` is honoured; a wait longer than 60 seconds is not slept through, and the error is raised with `retry_after` set. `402` is never retried. Neither are:

- a `503` without `Retry-After`. The API sends one when a search ran out of time ("Search took too long"), which fails the same way again;
- a `500` on `search_jobs`. Search takes its charge before it builds the response, so a `500` can follow a charge and a replay would bill the jobs again. `search_jobs` retries only `429` and a `503` with `Retry-After`, which come before any work.

Errors are raised as typed exceptions, importable from `llama_index.tools.jobo`, all deriving from `JoboError`:

| Exception | Status |
| --- | --- |
| `JoboValidationError` | 400 |
| `JoboAuthenticationError` | 401 |
| `JoboInsufficientCreditsError` | 402 (`top_up_url` when the API sends one) |
| `JoboPermissionError` | 403 |
| `JoboNotFoundError` | 404 |
| `JoboRateLimitError` | 429 (`retry_after`) |
| `JoboServerError` | 5xx |
| `JoboConnectionError` | no HTTP response |

An id that is not a UUID, or that the API says does not exist, comes back from `get_job` as a plain message. Any other `404` raises, so a wrong `base_url` is not read as "that job is gone".

## Connections

A `JoboToolSpec` keeps one HTTP connection pool for its sync calls, and one for its async calls per event loop, and reuses them for every call. Close them when you are done with `tool_spec.close()`, or `await tool_spec.aclose()` on the event loop the tools ran on. A closed tool spec can be used again: its next call opens a new pool. An `http_client` or `http_async_client` you pass in is yours: it is used as it is and never closed.

The package is typed (`py.typed`).

## Development

```bash
uv run -- pytest
```

The tests mock all HTTP and never touch the network.
