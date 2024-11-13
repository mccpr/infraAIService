import asyncio
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy
from fastapi import HTTPException

from infra_ai_service.model.model import EmbeddingOutput
from infra_ai_service.sdk import pgvector
from infra_ai_service.config.config import settings


async def create_embedding(content, os_version, name):
    try:
        # 使用线程池执行同步的嵌入计算
        url = f"{settings.PROXY_URL}/embeddings"
        headers = {
                "Content-Type": "application/json"
        }

        body = {
            "prompt": content,
            "model": "bge-large-en-v1.5",
            "encoding_format": "float"
        }
        embedding_vector = requests.post(url, headers=headers, json=body)

        # 检查返回类型是否为 ndarray，如果是，则转换为列表
        if isinstance(embedding_vector, numpy.ndarray):
            embedding_vector_list = embedding_vector.tolist()
        else:
            embedding_vector_list = embedding_vector  # 假设已经是列表

        # 从连接池获取连接
        async with pgvector.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO documents "
                    "(content, embedding, os_version, name) "
                    "VALUES (%s, %s, %s, %s) RETURNING id",
                    (content, embedding_vector_list, os_version, name),
                )
                point_id = (await cur.fetchone())[0]

        return EmbeddingOutput(
            id=str(point_id), embedding=embedding_vector_list
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing embedding: {e}"
        )
