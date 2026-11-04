from __future__ import annotations


class OculoStudioError(RuntimeError):
    """Base class for user-facing OculoStudio errors."""


class ValidationError(OculoStudioError):
    """Input or project state is invalid."""


class AssetError(OculoStudioError):
    """An asset cannot be inspected or materialized."""


class TimeGraphError(OculoStudioError):
    """A time-domain mapping is missing or invalid."""


class ProcessingError(OculoStudioError):
    """A processing operation failed."""


class DependencyError(OculoStudioError):
    """An optional runtime dependency is unavailable."""


class CancelledError(OculoStudioError):
    """A background job was cancelled by the user."""
