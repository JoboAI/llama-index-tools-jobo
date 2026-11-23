# Changelog

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
