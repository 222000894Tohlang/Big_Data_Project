
from pydantic import BaseModel, Field


class SpeciesPrediction(BaseModel):
    species: str
    confidence: float = Field(ge=0, le=1)


class PredictionTiming(BaseModel):
    preprocessing_ms: float = Field(ge=0)
    inference_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class PredictionResponse(BaseModel):
    model_name: str = "cnn"
    species: str
    confidence: float = Field(ge=0, le=1)
    top_predictions: list[SpeciesPrediction]
    timing: PredictionTiming
