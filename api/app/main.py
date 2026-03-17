import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.recognize import router as recognize_router
from app.routes.tajwid import router as tajwid_router
from app.routes.feedback import router as feedback_router
from app.core.model_loader import load_all_models

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_all_models()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(recognize_router)
app.include_router(feedback_router)
app.include_router(tajwid_router)


@app.get("/health")
def health():
    return {"status": "ok"}
