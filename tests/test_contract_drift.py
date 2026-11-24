"""
Drift guard against the shared connector filter contract.

`Jobo.Connectors/contracts/filters.json` is generated from the connector core
and lists every filter the Jobo connectors support. These tests fail when the
contract gains a filter or a value that `search_jobs` does not take, or does
not send.

The contract lives one directory above this package in the Jobo monorepo. The
published mirror of this directory has no contracts folder, so there the
module is skipped.
"""

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from llama_index.tools.jobo import JoboToolSpec, base
from llama_index.tools.jobo._format import FACET_FILTERS

CONTRACT_PATH = Path(__file__).resolve().parents[2] / "contracts" / "filters.json"

pytestmark = pytest.mark.skipif(
    not CONTRACT_PATH.is_file(),
    reason="filters.json is only present inside the Jobo.Connectors monorepo",
)

#: Contract filter keys whose `search_jobs` argument is named differently.
KEY_TO_ARGUMENT = {"q": "query"}
#: Contract filter keys `search_jobs` leaves out on purpose. `description_format`
#: only changes a job's `description`, and no search_jobs Document carries the
#: description: `get_job` is where the model reads it.
NOT_ARGUMENTS = frozenset({"description_format"})
#: `search_jobs` arguments that are paging, not filters.
PAGING = ("page", "page_size")

#: A sample value per contract `kind`, valid for any filter of that kind.
SAMPLE_BY_KIND: Dict[str, Any] = {
    "string": "sample",
    "number": 123,
    "boolean": False,
    "dateTime": "2026-11-01T00:00:00Z",
    "stringList": ["sample"],
}


def contract() -> Dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def search_filters() -> List[Dict[str, Any]]:
    return contract()["search_filters"]


def by_key() -> Dict[str, Dict[str, Any]]:
    return {descriptor["key"]: descriptor for descriptor in search_filters()}


def argument_for(key: str) -> str:
    return KEY_TO_ARGUMENT.get(key, key)


def sample_for(descriptor: Dict[str, Any]) -> Any:
    kind = descriptor["kind"]
    options: List[str] = descriptor.get("options") or []
    if kind == "enum":
        # The last option is never the server default, which is not sent.
        return options[-1]
    if kind == "enumList":
        return [options[0]] if options else ["sample"]
    return SAMPLE_BY_KIND[kind]


def signature_defaults() -> Dict[str, Any]:
    """Every argument of `search_jobs`, with its default."""
    parameters = inspect.signature(JoboToolSpec.search_jobs).parameters
    return {name: p.default for name, p in parameters.items() if name != "self"}


def search_parameters() -> Dict[str, Any]:
    """The filter arguments of `search_jobs`: all but the paging."""
    return {k: v for k, v in signature_defaults().items() if k not in PAGING}


def search_body(**arguments: Any) -> Dict[str, Any]:
    """The request body `search_jobs` would send for these arguments."""
    tool = JoboToolSpec(api_key="jbe_test_drift")
    body, _ = tool._search_request(**{**signature_defaults(), **arguments})
    return body


def test_contract_was_found_and_has_filters() -> None:
    assert len(search_filters()) >= 30


def test_search_jobs_takes_every_search_filter_in_the_contract() -> None:
    parameters = search_parameters()
    missing = sorted(
        descriptor["key"]
        for descriptor in search_filters()
        if descriptor["key"] not in NOT_ARGUMENTS
        and argument_for(descriptor["key"]) not in parameters
    )

    assert missing == [], f"search_jobs lacks contract filters: {missing}"


def test_search_jobs_has_no_filter_the_contract_does_not_know() -> None:
    keys = {argument_for(descriptor["key"]) for descriptor in search_filters()}

    assert sorted(set(search_parameters()) - keys) == []


def test_the_keys_named_here_are_real_contract_keys() -> None:
    keys = set(by_key())

    assert NOT_ARGUMENTS <= keys
    assert set(KEY_TO_ARGUMENT) <= keys


def test_every_search_filter_in_the_contract_reaches_the_request_body() -> None:
    """An argument that is accepted but never sent would be a silent no-op."""
    baseline = search_body()
    unsent: List[str] = []
    for descriptor in search_filters():
        key = descriptor["key"]
        if key in NOT_ARGUMENTS:
            continue
        if search_body(**{argument_for(key): sample_for(descriptor)}) == baseline:
            unsent.append(key)

    assert unsent == []


ENUMS: List[Tuple[str, Tuple[str, ...]]] = [
    ("work_models", base.WORK_MODELS),
    ("employment_types", base.EMPLOYMENT_TYPES),
    ("experience_levels", base.EXPERIENCE_LEVELS),
    ("company_categories", base.COMPANY_CATEGORIES),
    ("search_sorts", base.SORTS),
]


@pytest.mark.parametrize(("contract_enum", "values"), ENUMS)
def test_enum_values_match_the_contract(
    contract_enum: str, values: Tuple[str, ...]
) -> None:
    assert list(values) == contract()["enums"][contract_enum]


def test_search_filter_options_match_the_closed_value_sets() -> None:
    closed = {
        "work_model": base.WORK_MODELS,
        "employment_type": base.EMPLOYMENT_TYPES,
        "experience_level": base.EXPERIENCE_LEVELS,
        "company_categories": base.COMPANY_CATEGORIES,
        "exclude_company_categories": base.COMPANY_CATEGORIES,
        "sort": base.SORTS,
    }
    descriptors = by_key()

    for key, values in closed.items():
        assert list(values) == descriptors[key]["options"], key


def test_language_accepts_every_option_of_the_contract() -> None:
    """`language` stays open (the API takes names such as "English" too)."""
    options: List[str] = by_key()["language"]["options"]

    assert options == contract()["enums"]["languages"]
    assert search_body(language=options)["languages"] == options


def test_lookup_facets_are_the_dynamic_option_sources_of_the_contract() -> None:
    """
    `find_filter_values` lists every value set the contract loads at run time.

    `companies` is the one source it does not list: `find_companies` does.
    """
    sources = {
        descriptor["dynamicOptions"]
        for descriptor in search_filters()
        if "dynamicOptions" in descriptor
    }

    assert set(base.FILTER_FACETS) == sources - {"companies"}
    assert set(FACET_FILTERS) == set(base.FILTER_FACETS)


def test_each_lookup_facet_names_a_filter_search_jobs_takes() -> None:
    """`find_filter_values` tells the model which search_jobs filter to use."""
    parameters = search_parameters()
    descriptors = by_key()

    for facet, filter_name in FACET_FILTERS.items():
        assert filter_name in parameters, facet
        assert descriptors[filter_name]["dynamicOptions"] == facet
