from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import orjson

from app.config import settings
import app.logging  # noqa: F401 to set up logging
from app.routes.text_eval import router as text_router
from app.routes.image_eval import router as image_router


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content) -> bytes:
        return orjson.dumps(content)


app = FastAPI(title="Eval Microservice", default_response_class=ORJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.ALLOW_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/_healthz")
async def healthz():
    return {"ok": True}


app.include_router(text_router)
app.include_router(image_router)
