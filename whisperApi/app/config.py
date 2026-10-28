import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    whisper_api_keys: str = "sk-whisper-local-key"
    whisper_models: str = "tiny"
    whisper_default_model: str = "tiny"
    whisper_device: str = "cpu"
    whisper_compute_type: Optional[str] = None
    whisper_max_concurrency: int = 2
    whisper_max_queue: int = 4
    whisper_max_upload_mb: int = 25
    whisper_max_audio_sec: Optional[int] = None
    whisper_beam_size: int = 5
    whisper_vad: bool = True
    whisper_models_dir: str = "./models"
    port: int = 8000

    @property
    def api_keys_list(self) -> List[str]:
        return [k.strip() for k in self.whisper_api_keys.split(",") if k.strip()]

    @property
    def models_list(self) -> List[str]:
        return [m.strip() for m in self.whisper_models.split(",") if m.strip()]

    @property
    def effective_compute_type(self) -> str:
        if self.whisper_compute_type:
            return self.whisper_compute_type.strip()
        if self.whisper_device.lower() == "cuda":
            return "float16"
        return "int8"

    def resolve_model(self, model_name: str) -> str:
        if model_name == "whisper-1":
            return self.whisper_default_model
        return model_name

    def is_valid_model(self, model_name: str) -> bool:
        if model_name == "whisper-1":
            return True
        return model_name in self.models_list


settings = Settings()
