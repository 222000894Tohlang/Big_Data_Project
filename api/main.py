import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import predict_species
from schemas import PredictionResponse

from model_loader import load_model, get_device, get_class_names

from config import (
    MAX_UPLOAD_SIZE,
    ALLOWED_IMAGE_TYPES,
)

from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading wildlife CNN...")

    load_model()

    logger.info("Wildlife CNN ready")

    yield

    logger.info("Shutting down wildlife API")


app = FastAPI(
    title="Wildlife Species Classification API",
    description="REST API for WildlifeReID-10K species classification.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Wildlife Species Classification API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "wildlife-classification-api",
    }


@app.get("/health/model")
def model_health():
    try:
        load_model()

        return {
            "status": "ready",
            "model": "SimpleCNN",
            "device": str(get_device()),
            "num_classes": len(get_class_names()),
        }

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="CNN model is unavailable.",
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):

    start_time = time.perf_counter()

    try:
        # 1. Validate declared file type
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Only JPEG and PNG images are supported.",
            )

        # 2. Read uploaded file with size limit
        image_bytes = await file.read(MAX_UPLOAD_SIZE + 1)

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty.",
            )

        if len(image_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 10 MB upload limit.",
            )

        # 3. Run CNN inference
        prediction = predict_species(image_bytes)

        # 4. Record processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Prediction completed: species=%s, time=%.2fms",
            prediction["species"],
            processing_time_ms,
        )

        return prediction

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception("CNN prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again.",
        ) from None

    finally:
        await file.close()
