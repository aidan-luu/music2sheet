"""Dump the FastAPI app's OpenAPI schema to stdout.

CI runs this and diffs against the committed `api/openapi.json` to catch
silent schema drift (Agent A's contract must not change without a ticket).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as a plain script (`python api/scripts/dump_openapi.py`) even
# when the `api` package isn't installed onto sys.path by pip — the project
# build currently bypasses package selection (see pyproject.toml).
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
