from fastapi import FastAPI
from app.routes.recognize import router as recognize_router

app = FastAPI()

app.include_router(recognize_router)


@app.get("/health")
def health():
    return {"status": "ok"}
