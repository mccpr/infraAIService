
import requests
from loguru import logger

from infra_ai_service.model.model import EmbeddingOutput
from infra_ai_service.sdk import pgvector
from infra_ai_service.config.config import settings

def embedding(content):
    url = f"{settings.PROXY_URL}/embeddings"
    headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {settings.PROXY_TOKEN}"
    }
    body = {
        "prompt": content,
        "model": "bge-large-en-v1.5",
        "encoding_format": "float"
    }
    logger.info(f"embedding url: {url}  token: {settings.PROXY_TOKEN}")
    response = requests.post(url, headers=headers, json=body)
    logger.info(f"embedding response {response.embeddings}")
    return response.embeddings
