from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import inference_router

# Setup FastAPI app
app = FastAPI(title='Vision-Language Model Inference API')

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Include inference router
app.include_router(inference_router.router, prefix='/vlmodel')