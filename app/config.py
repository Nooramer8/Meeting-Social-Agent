from pathlib import Path
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    openai_api_key: str | None = None
    groq_api_key: str | None = None
    # groq: Whisper on GroqCloud (free tier). openai: gpt-4o-transcribe / whisper-1 on OpenAI.
    transcription_backend: str = 'groq'
    transcription_model: str = 'whisper-large-v3-turbo'
    # groq: Llama/Qwen/etc. on GroqCloud (free tier). openai: requires billable OpenAI quota.
    summary_backend: str = 'groq'
    # Groq models with json_schema: https://console.groq.com/docs/structured-outputs#supported-models
    summary_model: str = 'openai/gpt-oss-20b'
    default_language: str = 'auto'  # auto, ar, en, or any ISO-639-1 language code supported by transcription
    cloud_request_timeout_seconds: float = 180.0
    cloud_transcription_retries: int = 2
    cloud_transcription_retry_delay_seconds: float = 2.0

    @field_validator('groq_api_key', mode='before')
    @classmethod
    def _blank_groq_key(cls, value: object) -> str | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return str(value).strip()

    # Local/trained ASR pipeline
    # trained_transcription_backend: faster_whisper, transformers_whisper, remote, or disabled
    trained_transcription_backend: str = 'faster_whisper'
    trained_transcription_model_size: str = 'small'
    trained_asr_model_path: str = 'models/asr/whisper-ar-meetings'
    trained_asr_device: str = 'cpu'
    trained_asr_compute_type: str = 'int8'
    trained_asr_archive_url: str | None = None
    trained_asr_hf_repo_id: str | None = None
    trained_asr_hf_revision: str | None = None

    # Local/trained summarization pipeline
    # trained_summary_backend: rule_based, transformers_seq2seq, remote, or disabled
    trained_summary_backend: str = 'rule_based'
    trained_summary_model_path: str = 'models/summarizer/meeting-summary-ar'
    trained_summary_device: str = 'cpu'
    trained_summary_archive_url: str | None = None
    trained_summary_hf_repo_id: str | None = None
    trained_summary_hf_revision: str | None = None
    trained_remote_api_url: str | None = None
    huggingface_token: str | None = None

    # Which successful pipeline should populate the editable drafts: groq or trained.
    draft_source: str = 'groq'

    meta_graph_version: str = 'v25.0'
    facebook_page_id: str | None = None
    facebook_page_access_token: str | None = None
    instagram_user_id: str | None = None
    instagram_access_token: str | None = None
    public_base_url: str | None = None

    database_path: str = './storage/app.db'
    uploads_dir: str = './storage/uploads'
    generated_dir: str = './storage/generated'

    app_host: str = '0.0.0.0'
    app_port: int = 8000

    @property
    def uploads_path(self) -> Path:
        return Path(self.uploads_dir).resolve()

    @property
    def generated_path(self) -> Path:
        return Path(self.generated_dir).resolve()

@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    settings.generated_path.mkdir(parents=True, exist_ok=True)
    Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)
    return settings
