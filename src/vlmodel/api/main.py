from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import api.vlmodel_router as vlmodel_router

# Setup FastAPI app
api = FastAPI(title='Vision-Language Model Inference API')

# Enable CORSMiddleware
api.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Include inference router
api.include_router(vlmodel_router.router, prefix='/vlmodel')
