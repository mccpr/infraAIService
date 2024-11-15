
import requests
from loguru import logger

from infra_ai_service.model.model import EmbeddingOutput
from infra_ai_service.sdk import pgvector
from infra_ai_service.config.config import settings

def embedding(content):
    url = f"{settings.PROXY_URL}/embeddings"
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.PROXY_TOKEN}"
    }
    body = {
        "prompt": content,
        "model": "bge-large-en-v1.5",
        "encoding_format": "float"
    }
    logger.info(f"embedding url: {url}  headers: {headers}")
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        try:
            response_data = response.json()
            embeddings = response_data.get("embeddings")
            
            if embeddings is not None:
                return embeddings
            else:
                logger.error("No embeddings found in the response.")
                raise ValueError("No embeddings found in the response.")
        except ValueError as e:
            logger.error(f"Failed to parse the response: {e}")
            raise
    else:
        logger.error(f"Failed to get embeddings, status code: {response.status_code}")
        raise Exception(f"Error fetching embeddings: {response.status_code}")
