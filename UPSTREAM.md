# Contributing `llama-index-tools-jobo` upstream

This file is for maintainers of this folder. It is not part of the package and is not published to PyPI (the sdist includes only `llama_index/`).

Requirements below were read from a shallow clone of `run-llama/llama_index` at commit `99d7433` (2026-11-13, `main`). Re-read them before acting: they change.

## Read this first: upstream currently refuses new integration packages

`CONTRIBUTING.md`, section "What can you work on?":

> While we welcome contributions, **we are no longer accepting new integration packages** in this repository. New integrations should be maintained in their own repositories and published to PyPI independently. PRs that add a new `pyproject.toml` will be automatically closed.

`.github/workflows/close_new_integration_prs.yml` enforces it. It runs on every PR that touches a `pyproject.toml`, and when any file named `*/pyproject.toml` has status `added` it comments "we are no longer accepting new integration packages" and sets the PR to `closed`. This package's `pyproject.toml` is a new file, so a straight copy-in PR will be closed by the bot within moments of opening.

What that means for the plan "open a PR so the package is listed on LlamaHub":

- The in-repo PR route, as documented in the repo today, is closed for new packages. The steps below still describe exactly how it would be done, but expect the auto-close.
- The workflow's closing comment says to comment on the PR "if you believe this was closed in error (e.g., you are modifying an existing package)". A new package is not that case. Asking a maintainer first (for example on the project's Discord, linked from `CONTRIBUTING.md`) is the honest way to find out whether an exception or another listing route exists.
- The route CONTRIBUTING.md actually names is an independent package: its own repository, published to PyPI. Not confirmed from the files read: how, or whether, an independently published package gets listed on LlamaHub today. The `[tool.llamahub]` table in this `pyproject.toml` is kept because upstream's PR template asks for it.

## How the package is released instead

From its own public repository, `JoboAI/llama-index-tools-jobo`, which the monorepo's mirror workflow fills from this folder. `.github/workflows/release.yml` there publishes to PyPI with trusted publishing when a GitHub release `vX.Y.Z` is cut; the steps are in `Jobo.Connectors/RELEASING.md`, section 8. Bump the version in both `pyproject.toml` and `llama_index/tools/jobo/_client.py` (a unit test fails when they differ) and add it to `CHANGELOG.md`.

## Steps (if you attempt the in-repo PR anyway)

1. Fork `https://github.com/run-llama/llama_index` on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<your-account>/llama_index.git
   cd llama_index
   git checkout -b add-llama-index-tools-jobo
   ```

2. Copy this folder in, leaving out this file and local junk:

   ```bash
   rsync -a --exclude UPSTREAM.md --exclude .gitignore --exclude __pycache__ \
     /path/to/jobo/Jobo.Connectors/llamaindex/ \
     llama-index-integrations/tools/llama-index-tools-jobo/
   ```

   Neighbouring tool packages ship a `.gitignore` (starting with `llama_index/_static`); copy the one from `llama-index-tools-jina` instead of ours so the packages match.

3. The neighbours also commit a `uv.lock`. Generate it:

   ```bash
   cd llama-index-integrations/tools/llama-index-tools-jobo
   uv lock
   ```

4. Install the repo's tooling and run the checks (from `CONTRIBUTING.md`, "Quick Start Guide"):

   ```bash
   cd ../../..                       # repo root
   uv sync
   uv run pre-commit install
   uv run make format
   uv run make lint
   cd llama-index-integrations/tools/llama-index-tools-jobo
   uv run -- pytest
   ```

   The PR template checklist says: "I ran `uv run make format; uv run make lint` to appease the lint gods". `make lint` runs `pre-commit run` over the tracked files: pre-commit-hooks, `ruff` and `ruff-format`, `mypy`, `black` for docs and markdown code blocks (line length 79), `prettier`, `codespell`, `nb-clean` and `toml-sort-fix`. Several hooks only see files that git tracks, so `git add` the new folder before `make lint`.

5. Commit and push, then open the PR against `run-llama/llama_index:main` and fill in `.github/pull_request_template.md`.

## What upstream requires (quoted from the clone)

CONTRIBUTING.md, "Steps to Contribute" and "Development Guidelines":

- Fork, clone, branch, "Work on your feature or bugfix, ensuring you have unit tests covering your code", push, open a pull request.
- Tests: "We use `pytest` for testing." Run with `uv run -- pytest` in the package. "If you're integrating with a remote system, **mock** it to prevent test failures from external changes."
- Coverage: "By default, CI/CD will fail if test coverage is less than 50%". `.github/workflows/coverage_check.yml` sets `COV_FAIL_UNDER: 50`.
- Python: neighbouring packages declare `requires-python = ">=3.11,<4.0"`, and `.github/workflows/unit_test.yml` runs tests on Python 3.11 and 3.12 (and core on 3.14).
- AI use: CONTRIBUTING.md has a "How to Use AI when Contributing" section asking you to "highlight when and where you used AI to generate code, and explain how you verified and validated it". It also lists "secrets handling or security-related code" under things to avoid using AI for. This package was written with AI assistance and handles an API key: say so in the PR and have a human review `llama_index/tools/jobo/_client.py` closely.

PR template (`.github/pull_request_template.md`):

- "New Package? Did I fill in the `tool.llamahub` section in the `pyproject.toml` and provide a detailed README.md for my new integration or package?" Yes/No.
- "Version Bump? Did I bump the version in the `pyproject.toml` file of the package I am updating? (Except for the `llama-index-core` package)" The package starts at `0.1.0`; bump it on every later change.
- Type of change, "How Has This Been Tested?" ("Your pull-request will likely not be merged unless it is covered by some form of impactful unit testing."), and the suggested checklist (self-review, comments, docs, "I ran `uv run make format; uv run make lint`").

CI workflows that run on a PR: `lint.yml` (`uv run -- pre-commit run -a`), `unit_test.yml` and `coverage_check.yml` (via `llama-dev`, which was not part of the sparse checkout, so its exact behaviour is not confirmed), `build_package.yml` (`uv build`, then installs the wheel; on Ubuntu and Windows), `codeql.yml`, and `close_new_integration_prs.yml` (above).

Versions and publishing: `.github/workflows/publish_sub_package.yml` publishes "any [sub-package] whose local version is missing from PyPI" after a merge to `main`. So the version in `pyproject.toml` is the release. Check that `llama-index-tools-jobo` is free on PyPI and that Jobo owns it before opening a PR.

CLA: no contributor licence agreement is mentioned in `CONTRIBUTING.md`, the PR template or the `.github` folder of the clone. Not confirmed beyond those files: the sparse checkout does not include everything, and a CLA bot could be configured outside the repo.

## Differences from neighbouring packages, on purpose

- `authors` and `maintainers` are `Jobo <support@jobo.world>`; neighbours use a placeholder or a GitHub username. `[tool.llamahub.class_authors]` is `JoboToolSpec = "JoboAI"` (neighbours put a GitHub username there; confirm `JoboAI` is the account that should own it).
- `LICENSE` names Jobo AI. Neighbours carry "The MIT License, Copyright (c) Jerry Liu". Upstream may ask for its own text in the repo.
- `[project.urls]` (Homepage, Documentation) and `keywords` are set; neighbours leave them out.
- `httpx` is a direct dependency (the HTTP client is httpx, with a mock transport in tests). The neighbours' only runtime dependencies are their vendor SDK and `llama-index-core`.
- No `examples/` notebook: `contains_example = false`, as in the Tavily, Brave Search and Jina packages. No `BUILD` files and no `CHANGELOG.md`: none of the three neighbours has either (Tavily has an old changelog).
- Docstrings are written for the model that reads them, so they state which calls are billed. Keep that when editing.
