from pydantic_settings import BaseSettings, SettingsConfigDict
import os

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')

class Settings(BaseSettings):
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int
    SIMILARITY_THRESHOLD: float = 0.3 

    model_config = SettingsConfigDict(env_file=env_path, extra='ignore')

settings = Settings()

