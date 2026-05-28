import json
import os

REGISTRY_PATH = "prompts/registry.json"


def load_registry() -> dict:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_prompt(version: str = None) -> str:
    """
    Loads a prompt template by version.
    If version is None, loads the current version from registry.

    This is how we treat prompts like code — versioned, tracked,
    and switchable without touching the Python source.
    """
    registry = load_registry()
    version = version or registry["current_version"]

    if version not in registry["versions"]:
        raise ValueError(f"Prompt version '{version}' not found in registry. "
                         f"Available: {list(registry['versions'].keys())}")

    filepath = registry["versions"][version]["file"]

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    with open(filepath, encoding="utf-8") as f:
        return f.read()


def get_current_version() -> str:
    return load_registry()["current_version"]


def list_versions() -> list[dict]:
    registry = load_registry()
    return [
        {"version": v, **meta}
        for v, meta in registry["versions"].items()
    ]