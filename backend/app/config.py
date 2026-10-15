from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCPARSE_", env_file=".env")

    # Local folder standing in for the Snowflake stage during development.
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    max_upload_mb: int = 200


settings = Settings()
