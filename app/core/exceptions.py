"""
app/core/exceptions.py
Custom exceptions for clean error handling throughout the app.
"""


class AppError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, original: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.original = original

    def __str__(self):
        if self.original:
            return f"{self.message} (caused by: {self.original})"
        return self.message


class PDFExtractionError(AppError):
    """Raised when PDF text extraction fails or yields no text."""
    pass


class LLMServiceError(AppError):
    """Raised when a Groq API call fails."""
    pass


class JSONParseError(AppError):
    """Raised when LLM response cannot be parsed as valid JSON."""
    pass


class ReportGenerationError(AppError):
    """Raised when PDF report generation fails."""
    pass


class CriticalReviewError(AppError):
    """Raised when critical review generation or comparison fails."""
    pass
