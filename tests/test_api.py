from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():
    complaint = "I found an unauthorized charge on my credit card statement."

    response = client.post("/predict", json={"complaint": complaint})

    assert response.status_code == 200

    result = response.json()
    assert isinstance(result["predicted_category"], str)
    assert 0 <= result["confidence"] <= 1
    assert result["status"] in ["Auto Routed", "Needs Review"]


def test_empty_complaint():
    response = client.post("/predict", json={"complaint": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Complaint text cannot be empty."


def test_long_complaint():
    complaint = "A" * 10001

    response = client.post("/predict", json={"complaint": complaint})

    assert response.status_code == 400


def test_missing_complaint():
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_homepage():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
