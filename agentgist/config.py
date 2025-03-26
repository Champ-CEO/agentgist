import random
import sys
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class ModelProvider(str, Enum):
    GROQ = "groq"


@dataclass
class ModelConfig:
    name: str
    temperature: float
    provider: ModelProvider


LLAMA_3_3 = ModelConfig("llama-3.3-70b-versatile", 0.1, ModelProvider.GROQ)
DEEPSEEK_R1 = ModelConfig("deepseek-r1-distill-llama-70b", 0.2, ModelProvider.GROQ)


class Config:
    SEED = 42
    LOG_FILE = "app.log"

    class Model:
        DEFAULT = LLAMA_3_3  # For general tasks
        REPORT_WRITER = DEEPSEEK_R1  # For complex tasks

    class Preprocessing:
        EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        FILTER_POST_COUNT = 3


def seed_everything(seed: int = Config.SEED):
    random.seed(seed)


def configure_logging():
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "colorize": True,
                "format": "<green>{time:YYYY-MM-DD - HH:mm:ss}</green> | <level>{level}</level> | {message}",
            },
        ]
    }
    logger.configure(**config)
