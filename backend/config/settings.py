from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "SpectraGuard v2"
    app_version: str = "2.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    camera_index: int = 0
    baseline_path: str = "storage/baselines"
    log_path: str = "storage/logs"
    snapshot_path: str = "storage/snapshots"
    history_path: str = "storage/history"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
