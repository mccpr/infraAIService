from fastapi import FastAPI
from fastapi.responses import UJSONResponse

from infra_ai_service.api.router import api_router
from infra_ai_service.sdk.pgvector import setup_model_and_pool


def get_app() -> FastAPI:
    """
    Get FastAPI application.

    This is the main constructor of an application.

    :return: application.
    """
    app = FastAPI(
        title="FastAPI Starter Project",
        description="FastAPI Starter Project",
        version="1.0",
        docs_url="/api/docs/",
        redoc_url="/api/redoc/",
        openapi_url="/api/openapi.json",
        default_response_class=UJSONResponse,
    )

    app.include_router(router=api_router, prefix="/api/v1")

    @app.on_event("startup")
    async def startup_event():
        setup_model_and_pool()

    return app
