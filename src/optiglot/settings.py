import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class Settings:
    """
    Global settings for your run
    Attributes:
        host_url: The URL of the server to connect to.
        dspy_module: The DSPy module to use for the run if you want to compile a DSPy progam. Leave empty if you are not using DSPy.
    """

    host_url: str = os.getenv("HOST_URL", "http://localhost:8000")
    dspy_module = None


settings = Settings()


def configure(host_url=None, dspy_module=None):
    if host_url is not None:
        settings.host_url = host_url
    if dspy_module is not None:
        settings.dspy_module = dspy_module


def start():
    pass
