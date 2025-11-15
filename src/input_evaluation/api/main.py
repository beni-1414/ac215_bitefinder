from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import orjson

from api.config import settings
from api.logging import setup_logging

from api.routes.text_eval import router as text_router
from api.routes.image_eval import router as image_router

setup_logging()


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content) -> bytes:
        return orjson.dumps(content)


api = FastAPI(title="Eval Microservice", default_response_class=ORJSONResponse)

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


api.include_router(text_router)
api.include_router(image_router)
