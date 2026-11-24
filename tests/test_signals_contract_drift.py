"""
Drift guard against the shared hiring signals contract.

`Jobo.Connectors/contracts/signals.json` is generated from the connector core
and describes `GET /api/signals`: its filters, which of them the per-company
endpoint takes, the signal types, the seniority levels, the labels and the
page limits. These tests fail when this package no longer matches it, and when
the contract gains a filter `get_hiring_signals` neither takes nor names below
as left out on purpose.

The contract lives one directory above this package in the Jobo monorepo. The
published mirror of this directory has no contracts folder, so there the
module is skipped.
"""

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from llama_index.tools.jobo import JoboToolSpec, _signals

CONTRACT_PATH = Path(__file__).resolve().parents[2] / "contracts" / "signals.json"

pytestmark = pytest.mark.skipif(
    not CONTRACT_PATH.is_file(),
    reason="signals.json is only present inside the Jobo.Connectors monorepo",
)

#: Contract filters `get_hiring_signals` leaves out on purpose. They are
#: exclusions, thresholds that narrow one or two types, and the less used
#: company and location filters: a developer sets them better than a model.
#: `build_signals_request` takes every one of them.
LEFT_OFF_THE_TOOL = frozenset(
    {
        "exclude_domains",
        "categories",
        "founded_after",
        "technology_categories",
        "job_regions",
        "job_cities",
        "min_new_jobs",
        "min_days_open",
        "reposted_only",
    }
)
#: Contract filters the tool takes in another form: `company_id` reads one
#: company from `GET /api/companies/{id}/signals` instead of a list of ids.
ARGUMENT_FOR = {"company_ids": "company_id"}
#: Arguments that are not filters: the company in the path, and paging.
NOT_FILTERS = frozenset({"company_id", "limit", "cursor"})

COMPANY_ID = "6c1f0a9e-3b7d-4c55-8e21-5a4d9f0b7c33"

#: A sample value per contract `kind`, valid for any filter of that kind.
SAMPLE_BY_KIND: Dict[str, Any] = {
    "stringList": ["sample"],
    "boolean": True,
    "date": "2026-11-01",
    "dateTime": "2026-11-01T00:00:00Z",
}


def contract() -> Dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def contract_filters() -> List[Dict[str, Any]]:
    return contract()["filters"]


def by_key() -> Dict[str, Dict[str, Any]]:
    return {descriptor["key"]: descriptor for descriptor in contract_filters()}


def signature_defaults() -> Dict[str, Any]:
    """Every argument of `get_hiring_signals`, with its default."""
    parameters = inspect.signature(JoboToolSpec.get_hiring_signals).parameters
    return {name: p.default for name, p in parameters.items() if name != "self"}


def tool_filters() -> set:
    """The arguments of `get_hiring_signals` that are API filters."""
    return set(signature_defaults()) - NOT_FILTERS


def sample_for(descriptor: Dict[str, Any]) -> Any:
    kind = descriptor["kind"]
    if kind == "enumList":
        return [descriptor["options"][0]]
    if kind == "number":
        return descriptor["min"]
    return SAMPLE_BY_KIND[kind]


def query(**arguments: Any) -> Dict[str, Any]:
    """The query `get_hiring_signals` would send for these arguments."""
    tool = JoboToolSpec(api_key="jbe_test_drift")
    _, params, _ = tool._signals_request(**{**signature_defaults(), **arguments})
    return params


def test_contract_was_found_and_has_filters() -> None:
    assert len(contract_filters()) >= 20


@pytest.mark.parametrize(
    ("contract_enum", "values"),
    [
        ("signal_types", _signals.SIGNAL_TYPES),
        ("seniority", _signals.SIGNAL_SENIORITY),
        ("company_sizes", _signals.COMPANY_SIZES),
    ],
)
def test_enum_values_match_the_contract(contract_enum: str, values: Any) -> None:
    assert list(values) == contract()["enums"][contract_enum]


def test_the_closed_filters_check_against_the_contract_options() -> None:
    descriptors = by_key()

    assert list(_signals.SIGNAL_TYPES) == descriptors["types"]["options"]
    assert list(_signals.SIGNAL_SENIORITY) == descriptors["seniority"]["options"]
    assert list(_signals.COMPANY_SIZES) == descriptors["company_sizes"]["options"]


def test_every_type_has_the_dashboard_label() -> None:
    assert _signals.SIGNAL_TYPE_LABELS == contract()["type_labels"]
    assert list(_signals.SIGNAL_TYPE_LABELS) == list(_signals.SIGNAL_TYPES)


def test_page_limits_match_the_contract() -> None:
    assert contract()["limits"] == {
        "default_limit": _signals.SIGNAL_DEFAULT_LIMIT,
        "max_limit": _signals.SIGNAL_MAX_LIMIT,
        "max_list_values": _signals.SIGNAL_MAX_LIST_VALUES,
    }


def test_the_request_builder_knows_every_filter_and_its_kind() -> None:
    lists = {
        descriptor["key"]
        for descriptor in contract_filters()
        if descriptor["kind"] in ("stringList", "enumList")
    }

    assert list(_signals.SIGNAL_FILTER_KEYS) == [d["key"] for d in contract_filters()]
    assert lists == _signals.SIGNAL_LIST_FILTER_KEYS


def test_the_per_company_filters_match_the_contract() -> None:
    marked = [d["key"] for d in contract_filters() if d.get("company")]
    keys = list(_signals.COMPANY_SIGNAL_FILTER_KEYS)

    assert keys == contract()["company_filter_keys"]
    assert keys == marked


def test_the_tool_takes_every_filter_or_it_is_left_out_on_purpose() -> None:
    arguments = set(signature_defaults())
    missing = sorted(
        key
        for key in by_key()
        if key not in tool_filters()
        and key not in LEFT_OFF_THE_TOOL
        and ARGUMENT_FOR.get(key) not in arguments
    )

    assert missing == [], f"get_hiring_signals lacks contract filters: {missing}"


def test_the_tool_has_no_filter_the_contract_does_not_know() -> None:
    assert sorted(tool_filters() - set(by_key())) == []


def test_the_keys_named_here_are_real_contract_keys_and_off_the_tool() -> None:
    keys = set(by_key())

    assert LEFT_OFF_THE_TOOL <= keys
    assert set(ARGUMENT_FOR) <= keys
    assert LEFT_OFF_THE_TOOL.isdisjoint(tool_filters())
    assert set(ARGUMENT_FOR.values()) <= set(signature_defaults())


def test_every_filter_of_the_tool_reaches_the_query() -> None:
    """An argument that is accepted but never sent would be a silent no-op."""
    unsent: List[str] = []
    for descriptor in contract_filters():
        key = descriptor["key"]
        if key in tool_filters() and key not in query(**{key: sample_for(descriptor)}):
            unsent.append(key)

    assert unsent == []


def test_every_left_out_filter_still_reaches_the_query() -> None:
    """A developer can send what the tool leaves out."""
    descriptors = by_key()

    for key in sorted(LEFT_OFF_THE_TOOL):
        _, params = _signals.build_signals_request({key: sample_for(descriptors[key])})
        assert key in params, key


def test_company_id_refuses_exactly_what_its_endpoint_does_not_take() -> None:
    company_keys = set(contract()["company_filter_keys"])

    for descriptor in contract_filters():
        key = descriptor["key"]
        if key not in tool_filters():
            continue
        arguments = {key: sample_for(descriptor), "company_id": COMPANY_ID}
        if key in company_keys:
            assert key in query(**arguments), key
            continue
        with pytest.raises(ValueError, match="company_id reads one company"):
            query(**arguments)


def test_the_tool_cap_stays_within_the_api_limit() -> None:
    maximum = contract()["limits"]["max_limit"]

    assert JoboToolSpec(api_key="jbe_test_drift", max_signals=maximum)
    with pytest.raises(ValueError, match="max_signals"):
        JoboToolSpec(api_key="jbe_test_drift", max_signals=maximum + 1)
