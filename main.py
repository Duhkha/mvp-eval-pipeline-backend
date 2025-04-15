from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.router import api_router 
from app.models import loader 
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Loading models...")
    loader.startup_load_models()
    logger.info("Application startup: Model loading complete.")
    yield
    logger.info("Application shutdown.")


app = FastAPI(
    title="MVP Evaluation Pipeline API",
    description="Processes text snippets to identify met skill expectations for employees.",
    version="0.2.0",
    lifespan=lifespan
)

app.include_router(api_router)


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the MVP Evaluation Pipeline API!"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly from main.py...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)