"""
tests/test_api.py
──────────────────
Quick smoke-tests for the FastAPI endpoints.

Run with pytest (server does NOT need to be running — uses TestClient):
    pytest tests/test_api.py -v

Or run against a live server:
    python tests/test_api.py
"""

import sys
import os

# Make sure src/ imports resolve when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── TestClient approach (no running server needed) ────────────────────
from fastapi.testclient import TestClient

# Delay import so pytest collection works even if models aren't trained yet
try:
    from app import app
    client = TestClient(app)
    HAS_APP = True
except Exception as e:
    HAS_APP = False
    IMPORT_ERROR = str(e)


# ══════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════

def _skip_if_no_app():
    import pytest
    if not HAS_APP:
        pytest.skip(f"App import failed (models not trained?): {IMPORT_ERROR}")


# ══════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════

def test_root():
    _skip_if_no_app()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_health():
    _skip_if_no_app()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "backend" in data


def test_classes():
    _skip_if_no_app()
    resp = client.get("/classes")
    assert resp.status_code == 200
    classes = resp.json()["classes"]
    assert set(classes) == {"adhd", "aspergers", "depression", "ocd", "ptsd"}


def test_predict_depression():
    _skip_if_no_app()
    resp = client.post(
        "/predict",
        json={"text": "I feel completely hopeless and can't get out of bed anymore."},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Check all required fields are present
    assert "prediction" in data
    assert "confidence" in data
    assert "all_scores" in data
    assert "sentiment" in data
    assert "sentiment_scores" in data
    assert "severity" in data
    assert "clean_text" in data

    # Confidence should be a valid probability
    assert 0.0 <= data["confidence"] <= 1.0

    # Severity must be one of the three levels
    assert data["severity"] in ("low", "moderate", "high")

    # Sentiment must be a valid label
    assert data["sentiment"] in ("positive", "negative", "neutral")


def test_predict_short_text_rejected():
    _skip_if_no_app()
    resp = client.post("/predict", json={"text": "hi"})
    # FastAPI/Pydantic should reject text shorter than min_length=5
    assert resp.status_code == 422


def test_predict_missing_field():
    _skip_if_no_app()
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════════
# Manual runner (against live server)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import requests

    BASE = "http://localhost:8000"

    sample_texts = [
        "I feel anxious all the time and can't stop checking things.",
        "I can never focus on anything, my mind jumps from topic to topic constantly.",
        "I feel completely hopeless. Nothing makes me happy anymore.",
        "I keep having flashbacks from what happened to me last year.",
        "Social situations are extremely difficult for me to navigate.",
    ]

    print("\n" + "=" * 60)
    print("MindScope API — Manual Test")
    print("=" * 60)

    health = requests.get(f"{BASE}/health").json()
    print(f"\n[Health] status={health['status']}  backend={health['backend']}")

    for text in sample_texts:
        resp = requests.post(f"{BASE}/predict", json={"text": text})
        d    = resp.json()
        print(f"\n📝 Input    : {text[:60]}...")
        print(f"   Prediction: {d['prediction']}  (confidence: {d['confidence']:.3f})")
        print(f"   Sentiment : {d['sentiment']}  | Severity: {d['severity']}")
        print(f"   All scores: {d['all_scores']}")