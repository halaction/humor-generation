from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings for the humor agent."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-5-nano"


settings = Settings()
