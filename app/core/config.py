"""app/core/config.py"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    groq_api_key:  str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model:    str = "llama-3.3-70b-versatile"
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    max_tokens_section_extraction: int = 1500  # per chunk
    max_tokens_summary:            int = 2500  # full section-by-section summary
    max_tokens_critical_review:    int = 2200  # profile generation and pairwise comparison

    llm_max_retries: int   = 2
    llm_retry_delay: float = 2.0

    app_title:     str = "Research Paper Summarizer"
    app_icon:      str = "📄"
    report_author: str = "AI Research Assistant"

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.strip())


def get_settings() -> Settings:
    return Settings()
