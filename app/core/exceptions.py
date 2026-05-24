"""
Domain-level exception hierarchy.
All HTTPException mapping is done in the global exception handler in main.py.
"""


class VisionDxError(Exception):
    """Base exception for all application errors."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class NotFoundError(VisionDxError):
    def __init__(self, resource: str, identifier: str):
        super().__init__(f"{resource} '{identifier}' not found", "NOT_FOUND")


class DuplicateError(VisionDxError):
    def __init__(self, resource: str, field: str):
        super().__init__(f"{resource} with that {field} already exists", "DUPLICATE")


class AuthenticationError(VisionDxError):
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(detail, "AUTHENTICATION_ERROR")


class AuthorizationError(VisionDxError):
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(detail, "AUTHORIZATION_ERROR")


class ImageValidationError(VisionDxError):
    def __init__(self, detail: str):
        super().__init__(detail, "IMAGE_VALIDATION_ERROR")


class InferenceError(VisionDxError):
    def __init__(self, detail: str):
        super().__init__(detail, "INFERENCE_ERROR")


class StorageError(VisionDxError):
    def __init__(self, detail: str):
        super().__init__(detail, "STORAGE_ERROR")
