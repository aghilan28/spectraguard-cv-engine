"""Domain-specific exception mapping for CV processes."""

from .enums import CVErrorCode


class CVEngineError(Exception):
    """Base exception for all CV engine failures."""

    def __init__(self, message: str, code: CVErrorCode):
        super().__init__(f"[{code.value}] {message}")
        self.code = code
        self.message = message
