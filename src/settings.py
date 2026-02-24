from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    DEV: bool = False

    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_API_KEY: str = "sk-xxx"

    HF_TOKEN: str = "hf_xxx"
    HF_DATASET_REPO_ID: str = "org/name"


settings = Settings()
