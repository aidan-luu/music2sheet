"""Dump the FastAPI app's OpenAPI schema to stdout.

CI runs this and diffs against the committed `api/openapi.json` to catch
silent schema drift (Agent A's contract must not change without a ticket).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

# When invoked as a plain script from a worktree where `api` is not the
# pip-installed editable package (e.g. an agent worktree under
# .orchestrator/worktrees/<agent>/), add the repo root to sys.path so the
# local `api/` wins. This is a no-op in the canonical install where `api`
# already resolves.
if importlib.util.find_spec("api") is None:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from api.main import app  # noqa: E402


def main() -> int:
    schema = app.openapi()
    json.dump(schema, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
