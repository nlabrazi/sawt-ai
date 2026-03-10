from fastapi import FastAPI
from app.routes.recognize import router as recognize_router
from app.core.model_loader import load_all_models

app = FastAPI()


@app.on_event("startup")
def startup_event():
    load_all_models()


app.include_router(recognize_router)


@app.get("/health")
def health():
    return {"status": "ok"}
