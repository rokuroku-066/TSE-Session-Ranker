class SessionRankerError(Exception):
    """Base exception for the package."""


class DataValidationError(SessionRankerError):
    """Raised when a source violates the canonical data contract."""


class LeakageError(SessionRankerError):
    """Raised when timestamps or features could expose future information."""


class ArtifactError(SessionRankerError):
    """Raised when a model artifact is missing, corrupt, or incompatible."""

