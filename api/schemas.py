from pydantic import BaseModel


class PredictionResponse(BaseModel):
    species: str
    confidence: float