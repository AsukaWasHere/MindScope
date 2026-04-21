"""
api/schemas.py
───────────────
Pydantic models that define the shape of API request and response bodies.

FastAPI uses these to:
  - Automatically validate incoming JSON
  - Generate OpenAPI (Swagger) documentation at /docs
  - Serialize outgoing JSON responses

Keeping schemas in a separate file (not in app.py) makes them
easy to extend without touching the routing logic.
"""

from pydantic import BaseModel, Field
from typing import Literal


# ══════════════════════════════════════════════════════════════════════
# Request
# ══════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    """Body of POST /predict"""

    text: str = Field(
        ...,
        min_length=5,
        description="Raw post text to classify. No preprocessing required.",
        examples=["I feel anxious all the time and can't stop worrying."],
    )


# ══════════════════════════════════════════════════════════════════════
# Response
# ══════════════════════════════════════════════════════════════════════

class SentimentScores(BaseModel):
    neg:      float = Field(..., description="VADER negative score (0–1)")
    neu:      float = Field(..., description="VADER neutral score (0–1)")
    pos:      float = Field(..., description="VADER positive score (0–1)")
    compound: float = Field(..., description="VADER compound score (-1 to 1)")


class PredictResponse(BaseModel):
    """Body of the /predict response."""

    prediction: str = Field(
        ...,
        description="Predicted mental health condition (subreddit class).",
        examples=["depression"],
    )
    confidence: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Model probability for the predicted class.",
        examples=[0.87],
    )
    all_scores: dict[str, float] = Field(
        ...,
        description="Predicted probability for each of the 5 classes.",
        examples=[{"adhd": 0.04, "aspergers": 0.03,
                   "depression": 0.87, "ocd": 0.02, "ptsd": 0.04}],
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        ...,
        description="Overall sentiment label derived from VADER.",
        examples=["negative"],
    )
    sentiment_scores: SentimentScores = Field(
        ...,
        description="Full VADER polarity breakdown.",
    )
    severity: Literal["low", "moderate", "high"] = Field(
        ...,
        description=(
            "Heuristic severity estimate based on sentiment + prediction confidence. "
            "⚠️ NOT a clinical measure."
        ),
        examples=["high"],
    )
    clean_text: str = Field(
        ...,
        description="Preprocessed version of the input (for debugging / transparency).",
    )


# ══════════════════════════════════════════════════════════════════════
# Health-check response
# ══════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status:  str = Field(default="ok")
    backend: str = Field(..., description="'classical' or 'bert'")
    model:   str = Field(..., description="Name of the loaded model")


# ══════════════════════════════════════════════════════════════════════
# Error response
# ══════════════════════════════════════════════════════════════════════

class ErrorResponse(BaseModel):
    detail: str