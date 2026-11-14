"""Load API keys from the environment, Colab Secrets, or the git-ignored ``env.secrets`` file.

Lookup order: environment variable, then Colab ``userdata`` (when running on Colab), then
``env.secrets`` in the repo root. Values are never logged.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
SECRETS_FILE = _REPO_ROOT / "env.secrets"


def _from_file(name: str, path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None


def _from_colab(name: str) -> str | None:
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        value = userdata.get(name)
    except Exception:  # SecretNotFoundError / NotebookAccessError
        return None
    return str(value) if value else None


def get_secret(name: str, required: bool = True, path: Path = SECRETS_FILE) -> str | None:
    value = os.environ.get(name) or _from_colab(name) or _from_file(name, path)
    if required and not value:
        raise KeyError(f"Secret {name!r} not found in environment, Colab Secrets, or {path.name}.")
    return value
