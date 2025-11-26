from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import orjson

from api.config import settings
from api.logging import setup_logging

setup_logging()


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content) -> bytes:
        return orjson.dumps(content)


api = FastAPI(title="Orchestrator", default_response_class=ORJSONResponse)

api.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.ALLOW_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/_healthz")
async def healthz():
    return {"ok": True}


# Register orchestrator routes
from api.routes.orchestrator import router as orchestrator_router  # noqa: E402
from api.routes.rag import router as rag_router  # noqa: E402
from api.routes.rag_agent import router as rag_agent_router  # noqa: E402

api.include_router(orchestrator_router)
api.include_router(rag_router)
api.include_router(rag_agent_router)
