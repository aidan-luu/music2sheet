"""Shared pytest fixtures for the Sheet Sage 2 test suite.

Agent D owns this file. Other agents may request new fixtures by filing a
ticket under .orchestrator/tickets/ rather than editing directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def audio_fixtures_dir() -> Path:
    """Directory containing small audio clips + ground-truth annotations.

    Populated by Agent C once real fixture audio is procured (see
    tests/fixtures/README.md).
    """
    return Path(__file__).parent / "fixtures"
