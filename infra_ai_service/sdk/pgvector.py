import logging
from psycopg_pool import ConnectionPool

from infra_ai_service.common.utils import setup_database
from infra_ai_service.config.config import settings

logger = logging.getLogger(__name__)

pool = None


def setup_model_and_pool():
    global pool
    try:
        conn_str = (
            f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@"
            f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        pool = ConnectionPool(conn_str, min_size=5, max_size=20)
        logger.info(f"PostgreSQL connection pool created successfully.")

        setup_database(pool)
        logger.info("Database setup completed successfully.")

    except Exception as e:
        logger.error(f"Error setting up PostgreSQL connection pool: {e}", exc_info=True)
        raise 

def close_pool():
    """close pool"""
    global pool
    if pool:
        pool.close()
        logger.info("PostgreSQL connection pool closed.")
