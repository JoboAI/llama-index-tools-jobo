# Changelog

## 0.3.0

### Added

- **`get_hiring_signals`** and `aget_hiring_signals`: hiring signals over `GET /api/signals` and
  `GET /api/companies/{id}/signals`. A signal is one event at one company, detected from its job postings:
  a hiring surge, a first job in a country, a leadership search, a new technology, a slowdown.
  - Arguments: `types`, `domains`, `company_id`, `countries`, `company_sizes`, `industries`,
    `funding_stages`, `functions`, `roles`, `seniority`, `technologies`, `skills`, `job_countries`,
    `min_strength`, `detected_after`, `detected_before`, `created_after`, `exclude_agencies`, `limit` and
    `cursor`. With `company_id` the per-company endpoint is used, and a filter it does not take is refused
    with a message that names it.
  - Lists are sent comma-separated. A value that contains a comma, or a list of more than 100 values, is
    refused with `ValueError` before the request.
  - `limit` defaults to 5 and is capped at `max_signals` (default 25, at most 200). One call is one request:
    the answer gives the model the cursor of the next page.
  - Returns a summary document, one document per signal (company and domain, type, the dashboard's
    one-line summary, detection day, strength as a band and a number, evidence job ids) and a last document
    with what the call was billed. A type added to the API later is named, never an error.
  - A `402`, `403`, `429` or `400` comes with what the model can do about it. A sandbox key gets a message
    telling the model to stop, not an error.
- **`JoboToolSpec(include_hiring_signals=True, max_signals=...)`** adds the tool to that spec's
  `to_tool_list()`. It is off by default: every signal returned is billed at ten times the price of a job,
  and a signal read again is billed again.
- A contract drift test checks the signal types, the seniority levels, the labels, the page limits, the
  filters and the tool's arguments against the shared signals contract, and records the filters the tool
  leaves out on purpose.

### Changed

- **Hiring signals are not replayed after a `500`, a read timeout or a dropped connection**, like
  `search_jobs`: both take their charge before they build the response. Only `429` and a `503` with
  `Retry-After` are retried on them.

## 0.2.0

### Added

- **`search_jobs` takes the rest of the API's search filters**: `job_subcategories`, `exclude_industries`,
  `exclude_company_categories`, `exclude_job_categories`, `exclude_job_subcategories`,
  `exclude_occupation_ids`, `discovered_after`, `discovered_before` and `search_description`.
  `find_filter_values("job_subcategories")` told the model to pass its values in a `job_subcategories`
  filter that `search_jobs` did not have. `description_format` is still not an argument: search results
  carry no description. Pass `search_jobs` arguments by name: the new ones sit beside the filters they
  belong to, so the position of the later arguments changed.
- **`JoboToolSpec.close()` and `aclose()`.** The tool spec keeps one connection pool for sync calls and one
  per event loop for async calls, instead of opening a new `httpx` client for every attempt.
- A contract drift test checks the fixed value sets, the lookup facets and the `search_jobs` arguments
  against the shared connector filter contract, inside the Jobo connectors monorepo.
- `py.typed`: the package is marked as typed.

### Changed

- **The lowest supported `llama-index-core` is 0.13.1** (was 0.13.0). In 0.13.0, `FunctionTool` turned a
  returned list of `Document`s into `str(list)`, so the model read object dumps instead of the job text.
  0.13.1 added the branch that hands the model each `Document`'s content.
- `min_salary_usd` and `max_salary_usd` below 0 raise `ValueError` before any request.

### Fixed

- **`search_jobs` is not replayed after a `500`.** `POST /api/jobs/search` bills every job it returns and
  takes the charge before it builds the response, so a `500` can follow a charge and a retry billed the
  jobs again. Search now retries only `429` and a `503` with `Retry-After`.
- **A `503` without `Retry-After` is not retried**, on any call. The API sends it when a search ran out of
  time, which fails the same way again.
- The README's agent example runs as a script: `agent.run()` is awaited inside `asyncio.run(main())`.
- `[tool.mypy] python_version` is 3.11, the lowest Python the package supports.

## 0.1.0

First release: the Jobo jobs API as a LlamaIndex tool spec.

### Added

- **`JoboToolSpec`**, with five tools and an async twin of each:
  - `search_jobs`: job search over `POST /api/jobs/search` with the main filters, including the exclude
    forms of skills and companies. `page_size` defaults to 5 and is capped at `max_page_size` (default 25),
    because search is billed per job returned.
  - `get_job`: one listing by id, as readable prose. Free.
  - `find_companies`, `find_filter_values`, `find_locations`: turn a name into the ids and values a search
    filter takes. Free.
- Every tool returns `List[Document]`; a search returns a summary document, then one document per job whose
  `doc_id` is the Jobo job id.
- Typed errors deriving from `JoboError`, with retries on `429`, `500` and `503` that honour `Retry-After`.
- The API key is read from `JOBO_API_KEY` or passed as `api_key`, and is only ever sent in the `X-Api-Key`
  header.
