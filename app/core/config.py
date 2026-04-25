"""app/core/config.py"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    groq_api_key:   str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model:     str = "llama-3.3-70b-versatile"
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    max_tokens_section_extraction: int = 1500   # per chunk
    max_tokens_summary:            int = 2500   # full section-by-section summary

    # Raised from 2200 → 3200 so pairwise JSON with 8 dimensions is never
    # truncated mid-object.  critical_review_service passes MAX_PAIRWISE_TOKENS
    # / MAX_PROFILE_TOKENS directly; this field is a conservative fallback.
    max_tokens_critical_review: int = 3200

    llm_max_retries: int   = 2
    llm_retry_delay: float = 2.0

    # Pairwise comparison mode.
    # False (default) = fully local/deterministic, no extra LLM call.
    # True  = attempt an LLM-assisted pairwise call; auto-falls back to
    #         local if the provider returns rate-limit / quota / size errors.
    use_llm_for_pairwise: bool = field(
        default_factory=lambda: os.getenv("USE_LLM_FOR_PAIRWISE", "false").strip().lower()
        in ("1", "true", "yes")
    )

    app_title:     str = "Research Paper Summarizer"
    app_icon:      str = "📄"
    report_author: str = "AI Research Assistant"

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.strip())


def get_settings() -> Settings:
    return Settings()