from typing import Any, Dict, Optional
from fastapi import HTTPException, status

class BaseAppException(HTTPException):
    def __init__(
        self,
        status_code: int,
        detail: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)

class AuthenticationError(BaseAppException):
    def __init__(self, detail: str = "Could not validate credentials") -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

class ForbiddenError(BaseAppException):
    def __init__(self, detail: str = "Not enough permissions") -> None:
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)

class NotFoundError(BaseAppException):
    def __init__(self, detail: str = "Resource not found") -> None:
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

class ValidationError(BaseAppException):
    def __init__(self, detail: str = "Invalid input data") -> None:
        super().__init__(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)

class ConflictError(BaseAppException):
    def __init__(self, detail: str = "Resource already exists") -> None:
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)

class LLMServiceError(BaseAppException):
    """Raised when an upstream LLM provider fails (API error, timeout, invalid response)."""
    def __init__(self, detail: str = "LLM service encountered an error") -> None:
        super().__init__(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)

class RetrievalError(BaseAppException):
    """Raised when vector similarity search or chunk retrieval fails."""
    def __init__(self, detail: str = "Document retrieval failed") -> None:
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail
        )
