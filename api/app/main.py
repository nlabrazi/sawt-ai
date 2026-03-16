from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.recognize import router as recognize_router
from app.routes.tajwid import router as tajwid_router
from app.routes.feedback import router as feedback_router
from app.core.model_loader import load_all_models

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    load_all_models()


app.include_router(recognize_router)
app.include_router(feedback_router)
app.include_router(tajwid_router)


@app.get("/health")
def health():
    return {"status": "ok"}
