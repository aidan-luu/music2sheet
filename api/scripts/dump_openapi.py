"""Dump the FastAPI app's OpenAPI schema to stdout.

CI runs this and diffs against the committed `api/openapi.json` to catch
silent schema drift (Agent A's contract must not change without a ticket).
"""

from __future__ import annotations

import json
import sys

from api.main import app


def main() -> int:
    schema = app.openapi()
    json.dump(schema, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
