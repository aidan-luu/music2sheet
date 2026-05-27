"""ML inference entry point.

This is the public surface that Agent B's FastAPI service imports. Keep
:func:`transcribe` signature-stable across PRs; if it must change, file a
ticket to Agent B before merging.
"""

from __future__ import annotations

from ml.inference.pipeline import transcribe

__all__ = ["transcribe"]
