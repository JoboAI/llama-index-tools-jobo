"""
Render Jobo API objects as text a model can read and cite.

Pure functions over the API's JSON. A field the posting does not state is left
out of the text and of the metadata: nothing here fills a gap with a guess.
"""

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

_PERIODS = {
    "hourly": "per hour",
    "daily": "per day",
    "weekly": "per week",
    "monthly": "per month",
    "yearly": "per year",
    "per-diem": "per diem",
}

#: The search filter each lookup facet's values go into.
FACET_FILTERS = {
    "sources": "sources",
    "industries": "industries",
    "skills": "skills",
    "company_categories": "company_categories",
    "job_categories": "job_categories",
    "job_subcategories": "job_subcategories",
    "occupations": "occupation_ids",
}
_NAMED_FACETS = frozenset({"job_categories", "job_subcategories"})
_WORD = re.compile(r"[a-z0-9]+")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> List[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _strings(value: Any) -> List[str]:
    return [t for t in (_text(item) for item in _list(value)) if t]


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _money(value: float) -> str:
    return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"


def _day(value: Any) -> str:
    return _text(value)[:10]


def _count(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return ""
    return f"{value:,} job{'' if value == 1 else 's'}"


def job_places(job: Mapping[str, Any]) -> List[str]:
    """Every distinct place the job lists, in the order the API returned them."""
    places: List[str] = []
    for location in _list(job.get("locations")):
        loc = _map(location)
        label = _text(loc.get("location")) or ", ".join(
            p for p in (_text(loc.get(k)) for k in ("city", "region", "country")) if p
        )
        if label and label not in places:
            places.append(label)
    return places


def format_location(job: Mapping[str, Any]) -> str:
    """Render the job's location as one short phrase."""
    places = job_places(job)
    if not places:
        return "Remote" if job.get("workplace_type") == "Remote" else "Not stated"
    if len(places) == 1:
        return places[0]
    return f"{places[0]} (+{len(places) - 1} more)"


def format_compensation(job: Mapping[str, Any]) -> str:
    """Render compensation as published, or an empty string when there is none."""
    comp = _map(job.get("compensation"))
    low, high = _number(comp.get("min")), _number(comp.get("max"))
    if low is None and high is None:
        return ""
    if low is not None and high is not None:
        amount = _money(low) if low == high else f"{_money(low)}-{_money(high)}"
    elif low is not None:
        amount = f"from {_money(low)}"
    else:
        amount = f"up to {_money(high or 0)}"
    currency = _text(comp.get("currency"))
    period = _text(comp.get("period"))
    return " ".join(p for p in (currency, amount, _PERIODS.get(period, period)) if p)


def _job_url(job: Mapping[str, Any]) -> str:
    return _text(job.get("listing_url")) or _text(job.get("apply_url"))


def format_job_summary(job: Mapping[str, Any], index: int) -> str:
    """Render one job as a compact, citeable block for a search result."""
    title = _text(job.get("title")) or "Untitled job"
    company = _text(_map(job.get("company")).get("name"))
    lines = [f"{index}. {title} - {company}" if company else f"{index}. {title}"]
    facts = [f"Location: {format_location(job)}"]
    for label, key in (
        ("Work model", "workplace_type"),
        ("Employment", "employment_type"),
        ("Level", "experience_level"),
    ):
        value = _text(job.get(key))
        if value:
            facts.append(f"{label}: {value}")
    lines.append(" | ".join(facts))
    pay = format_compensation(job)
    if pay:
        lines.append(f"Salary: {pay}")
    if _day(job.get("date_posted")):
        lines.append(f"Posted: {_day(job.get('date_posted'))}")
    elif _day(job.get("created_at")):
        lines.append(f"First indexed: {_day(job.get('created_at'))}")
    if _job_url(job):
        lines.append(f"URL: {_job_url(job)}")
    lines.append(f"ID: {_text(job.get('id'))}")
    return "\n".join([lines[0], *(f"   {line}" for line in lines[1:])])


def _bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _qualification_items(group: Any) -> List[str]:
    group_map = _map(group)
    names = [
        n
        for n in (_text(_map(s).get("name")) for s in _list(group_map.get("skills")))
        if n
    ]
    return [
        *names,
        *_strings(group_map.get("education")),
        *_strings(group_map.get("certifications")),
    ]


def format_job_document(
    job: Mapping[str, Any], include_description: bool = True
) -> str:
    """Flatten a job into self-contained prose: structured fields, then description."""
    title = _text(job.get("title")) or "Untitled job"
    company = _map(job.get("company"))
    company_name = _text(company.get("name"))
    sections: List[str] = [f"# {title}"]

    facts: List[str] = []
    if company_name:
        facts.append(f"Company: {company_name}")
    facts.append(f"Location: {format_location(job)}")
    occupation = _map(job.get("occupation"))
    if _text(occupation.get("name")):
        path = " > ".join(
            p
            for p in (
                _text(occupation.get("category")),
                _text(occupation.get("subcategory")),
            )
            if p
        )
        facts.append(
            f"Occupation: {_text(occupation.get('name'))}"
            + (f" ({path})" if path else "")
        )
    for label, key in (
        ("Work model", "workplace_type"),
        ("Employment", "employment_type"),
        ("Level", "experience_level"),
    ):
        if _text(job.get(key)):
            facts.append(f"{label}: {_text(job.get(key))}")
    if format_compensation(job):
        facts.append(f"Compensation: {format_compensation(job)}")
    if _day(job.get("date_posted")):
        facts.append(f"Posted: {_day(job.get('date_posted'))}")
    if _day(job.get("valid_through")):
        facts.append(f"Closes: {_day(job.get('valid_through'))}")
    if _strings(company.get("industries")):
        facts.append(f"Industry: {', '.join(_strings(company.get('industries')))}")
    sections.append("\n".join(facts))

    if _text(company.get("summary")) and company_name:
        sections.append(f"## About {company_name}\n{_text(company.get('summary'))}")
    if _text(job.get("summary")):
        sections.append(f"## Summary\n{_text(job.get('summary'))}")
    if _strings(job.get("responsibilities")):
        sections.append(
            f"## Responsibilities\n{_bullets(_strings(job.get('responsibilities')))}"
        )
    qualifications = _map(job.get("qualifications"))
    for label, key in (
        ("Required qualifications", "must_have"),
        ("Preferred qualifications", "preferred"),
    ):
        items = _qualification_items(qualifications.get(key))
        if items:
            sections.append(f"## {label}\n{_bullets(items)}")
    if _strings(job.get("benefits")):
        sections.append(f"## Benefits\n{_bullets(_strings(job.get('benefits')))}")
    places = job_places(job)
    if len(places) > 1:
        sections.append(f"## All locations\n{_bullets(places)}")
    if include_description and _text(job.get("description")):
        sections.append(f"## Description\n{_text(job.get('description'))}")

    links: List[str] = []
    listing, apply = _text(job.get("listing_url")), _text(job.get("apply_url"))
    if listing:
        links.append(f"Listing: {listing}")
    if apply and apply != listing:
        links.append(f"Apply: {apply}")
    if _text(job.get("source")):
        links.append(f"ATS: {_text(job.get('source'))}")
    links.append(f"Job ID: {_text(job.get('id'))}")
    sections.append("\n".join(links))
    return "\n\n".join(sections)


def job_metadata(job: Mapping[str, Any]) -> Dict[str, Any]:
    """Flat Document metadata for a job. Keys without a value are omitted."""
    company = _map(job.get("company"))
    comp = _map(job.get("compensation"))
    candidates: Dict[str, Any] = {
        "id": _text(job.get("id")),
        "url": _job_url(job),
        "apply_url": _text(job.get("apply_url")),
        "title": _text(job.get("title")),
        "company": _text(company.get("name")),
        "company_id": _text(company.get("id")),
        "location": format_location(job) if job_places(job) else "",
        "workplace_type": _text(job.get("workplace_type")),
        "employment_type": _text(job.get("employment_type")),
        "experience_level": _text(job.get("experience_level")),
        "salary": format_compensation(job),
        "salary_min": _number(comp.get("min")),
        "salary_max": _number(comp.get("max")),
        "salary_currency": _text(comp.get("currency")),
        "salary_period": _text(comp.get("period")),
        "date_posted": _text(job.get("date_posted")),
        "created_at": _text(job.get("created_at")),
        "updated_at": _text(job.get("updated_at")),
        "valid_through": _text(job.get("valid_through")),
        "ats": _text(job.get("source")),
    }
    return {k: v for k, v in candidates.items() if v not in ("", None)}


def search_notes(payload: Mapping[str, Any]) -> List[str]:
    """Non-fatal warnings of a search response, and close company matches."""
    notes = [
        f"Note: {t}"
        for t in (_text(_map(w).get("message")) for w in _list(payload.get("warnings")))
        if t
    ]
    companies = _map(_map(payload.get("filters")).get("companies"))
    for entry in _list(companies.get("suggestions")):
        entry_map = _map(entry)
        rendered = []
        for option in _list(entry_map.get("companies"))[:3]:
            option_map = _map(option)
            name = _text(option_map.get("name"))
            if not name:
                continue
            ids = ", ".join(_strings(option_map.get("ids")))
            domain = _text(option_map.get("domain"))
            extra = "; ".join(p for p in (domain, f"ids: {ids}" if ids else "") if p)
            rendered.append(f"{name} ({extra})" if extra else name)
        if rendered:
            notes.append(
                f'Closest companies to "{_text(entry_map.get("query"))}": '
                + " | ".join(rendered)
            )
    return notes


def search_header(payload: Mapping[str, Any], shown: int) -> str:
    """The one-line summary above a page of search results."""
    total, page, pages = (
        payload.get("total"),
        payload.get("page"),
        payload.get("total_pages"),
    )
    header = f"Showing {shown} job{'' if shown == 1 else 's'}"
    if isinstance(total, int):
        header = f"{total:,} matching job{'' if total == 1 else 's'}; showing {shown}"
    if isinstance(page, int) and isinstance(pages, int):
        header += f" (page {page} of {pages:,})"
        if page < pages:
            header += f". More results are available: request page {page + 1}"
    return header + "."


EMPTY_SEARCH = (
    "No jobs matched. Relax a filter or broaden the query. Enum filters take "
    "lowercase keys, and an unrecognised filter value matches nothing."
)


def format_company_suggestions(
    query: str, suggestions: Sequence[Mapping[str, Any]]
) -> str:
    """Render company lookups: one line per company with every id to filter on."""
    if not suggestions:
        return (
            f'No company matched "{query}". Lookups match the start of a company '
            "name or website domain and need at least 2 characters: try a shorter "
            "or different spelling, or the company's domain."
        )
    lines: List[str] = []
    for index, s in enumerate(suggestions, start=1):
        facts = [f for f in (_text(s.get("domain")), _text(s.get("country"))) if f]
        line = f"{index}. {_text(s.get('name')) or 'Unnamed company'}"
        if facts:
            line += f" ({', '.join(facts)})"
        if _count(s.get("job_count")):
            line += f" - {_count(s.get('job_count'))} live"
        ids = _strings(s.get("ids")) or _strings([s.get("id")])
        lines.append(f"{line}\n   company ids: {', '.join(ids)}")
    header = (
        f'{len(suggestions)} match(es) for "{query}". To search one company\'s jobs, '
        "pass all of its ids as the `companies` filter of search_jobs."
    )
    return "\n\n".join([header, "\n".join(lines)])


def _filter_hint(facet: str) -> str:
    name = FACET_FILTERS.get(facet, facet)
    what = "the ids" if facet == "occupations" else "the values exactly as written"
    return f"Pass {what} in the `{name}` filter of search_jobs."


def format_filter_values(
    facet: str, query: str, options: Sequence[Mapping[str, Any]]
) -> str:
    """Render one facet's values, each with its live job count."""
    if not options:
        if query:
            return (
                f'No `{facet}` value matches "{query}". The lookup matches the '
                "start of a value: try a shorter spelling, or leave the query "
                "out to see the most common values."
            )
        return f"The API lists no `{facet}` values at the moment."
    lines: List[str] = []
    for index, option in enumerate(options, start=1):
        key = _text(option.get("key"))
        name = _text(option.get("name"))
        parent = _text(option.get("parent"))
        if facet == "occupations":
            line = f"{name or key} - id: {key}"
            if parent:
                line += f" (subcategory: {parent})"
        else:
            line = (name or key) if facet in _NAMED_FACETS else key
            if parent:
                line += f" (category: {parent})"
        count = _count(option.get("count"))
        lines.append(f"{index}. {line}{f' - {count}' if count else ''}")
    matching = f' matching "{query}"' if query else ""
    header = (
        f"{len(options)} `{facet}` value(s){matching}, most jobs first. "
        f"{_filter_hint(facet)}"
    )
    return "\n\n".join([header, "\n".join(lines)])


def flatten_taxonomy(data: Any) -> List[Dict[str, str]]:
    """`GET /api/occupations` as one row per occupation."""
    rows: List[Dict[str, str]] = []
    for category in _list(_map(data).get("categories")):
        cat = _map(category)
        for subcategory in _list(cat.get("subcategories")):
            sub = _map(subcategory)
            for occupation in _list(sub.get("occupations")):
                occ = _map(occupation)
                oid, name = occ.get("id"), occ.get("name")
                if isinstance(oid, str) and isinstance(name, str):
                    rows.append(
                        {
                            "id": oid,
                            "name": name,
                            "category": _text(cat.get("name")),
                            "subcategory": _text(sub.get("name")),
                        }
                    )
    return rows


def match_occupations(
    rows: Sequence[Dict[str, str]], query: str, limit: int
) -> List[Dict[str, str]]:
    """Occupations whose name holds every word of the query, best match first."""
    needle = query.lower()
    words = needle.split()
    ranked: List[Any] = []
    for index, row in enumerate(rows):
        name = row["name"].lower()
        name_words = set(_WORD.findall(name))
        if name == needle:
            rank = 0
        elif all(w in name_words for w in words):
            rank = 1
        elif name.startswith(needle):
            rank = 2
        elif all(w in name for w in words):
            rank = 3
        elif all(
            w in f"{name} {row['subcategory']} {row['category']}".lower() for w in words
        ):
            rank = 4
        else:
            continue
        ranked.append((rank, index, row))
    ranked.sort(key=lambda entry: entry[:2])
    return [row for _, _, row in ranked[:limit]]


def format_occupation_matches(
    query: str, occupations: Sequence[Mapping[str, Any]]
) -> str:
    """Render occupations found by name: the name, its id and where it sits."""
    if not occupations:
        return (
            f'No occupation matches "{query}". Try one general word from the '
            'job title, e.g. "nurse" or "developer".'
        )
    lines: List[str] = []
    for index, o in enumerate(occupations, start=1):
        path = " > ".join(
            p for p in (_text(o.get("category")), _text(o.get("subcategory"))) if p
        )
        line = f"{index}. {_text(o.get('name'))} - id: {_text(o.get('id'))}"
        lines.append(f"{line} ({path})" if path else line)
    header = f'{len(occupations)} occupation(s) matching "{query}". {_filter_hint("occupations")}'
    return "\n\n".join([header, "\n".join(lines)])


def format_location_suggestions(
    query: str, suggestions: Sequence[Mapping[str, Any]]
) -> str:
    """Render place lookups: one line per place, as the search filter takes it."""
    labels: List[str] = []
    for s in suggestions:
        label = _text(s.get("display_name"))
        if not label:
            parts: List[str] = []
            for key in ("city", "region", "country"):
                part = _text(s.get(key))
                if part and part not in parts:
                    parts.append(part)
            label = ", ".join(parts)
        if label and label not in labels:
            labels.append(label)
    if not labels:
        return (
            f'No place matched "{query}". Lookups need at least 2 characters: '
            "try a different spelling. search_jobs also takes a free-form "
            "`location` and resolves it itself."
        )
    header = (
        f'{len(labels)} place(s) matching "{query}". Pass one, exactly as '
        "written, as the `location` filter of search_jobs."
    )
    return "\n\n".join(
        [header, "\n".join(f"{i}. {label}" for i, label in enumerate(labels, start=1))]
    )
