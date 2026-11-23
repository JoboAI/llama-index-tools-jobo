"""The version is written twice. A release publishes one and sends the other."""

import tomllib
from pathlib import Path

from llama_index.tools.jobo import _client


def test_the_user_agent_version_is_the_package_version() -> None:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]

    assert _client.__version__ == declared
    assert _client.USER_AGENT == f"llama-index-tools-jobo/{declared}"
