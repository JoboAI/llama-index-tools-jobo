# Changelog

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
