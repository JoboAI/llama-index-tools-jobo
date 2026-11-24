"""The version is written twice. A release publishes one and sends the other."""

import tomllib
from pathlib import Path

from llama_index.tools.jobo import _client

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_the_user_agent_version_is_the_package_version() -> None:
    declared = tomllib.loads(PYPROJECT.read_text())["project"]["version"]

    assert _client.__version__ == declared
    assert _client.USER_AGENT == f"llama-index-tools-jobo/{declared}"


def test_the_package_is_marked_typed() -> None:
    assert Path(_client.__file__).with_name("py.typed").is_file()


def test_mypy_checks_the_python_the_package_requires() -> None:
    project = tomllib.loads(PYPROJECT.read_text())
    floor = project["project"]["requires-python"].split(",")[0].lstrip(">=")

    assert project["tool"]["mypy"]["python_version"] == floor
