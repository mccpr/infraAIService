import requests
from fastapi import HTTPException

from infra_ai_service.model.model import EmbeddingOutput
from infra_ai_service.sdk import pgvector
from infra_ai_service.config.config import settings


async def create_embedding(content, os_version, name):
    try:
        url = f"{settings.PROXY_URL}/embeddings"
        headers = {
                "Content-Type": "application/json"
        }

        body = {
            "prompt": content,
            "model": "bge-large-en-v1.5",
            "encoding_format": "float"
        }
        response = requests.post(url, headers=headers, json=body)

        async with pgvector.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO documents "
                    "(content, embedding, os_version, name) "
                    "VALUES (%s, %s, %s, %s) RETURNING id",
                    (content, response.embeddings, os_version, name),
                )
                point_id = (await cur.fetchone())[0]

        return EmbeddingOutput(
            id=str(point_id), embedding=response.embeddings
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing embedding: {e}"
        )
