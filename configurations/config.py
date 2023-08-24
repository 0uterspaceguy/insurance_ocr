from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "OCR API"
    triton_port: int
    triton_ip: str
    log_level: str
    fastapi_port: int


    class Config:
        env_file = ".env"


config = Settings()
