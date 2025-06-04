from pathlib import Path
from pydantic_settings import BaseSettings

class ApiSecrets(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    # HUGGINGFACEHUB_API_TOKEN: str
    GOOGLE_API_KEY: str
    LANGCHAIN_API_KEY: str
    LANGCHAIN_PROJECT: str
    LANGCHAIN_TRACING_V2: bool
    # OPENAI_API_KEY: str
    class Config:
        env_file = ".env"
        extra = "ignore"  

ai_api_secrets = ApiSecrets()