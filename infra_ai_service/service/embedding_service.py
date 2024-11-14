import logging

from fastapi import HTTPException

from infra_ai_service.model.model import EmbeddingOutput
from infra_ai_service.sdk import pgvector,ai_proxy

logger = logging.getLogger(__name__)

async def create_embedding(content, os_version, name):
    try:
        response = ai_proxy.embedding(content)
        logger.info("embedding response %s", response.embeddings)
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
