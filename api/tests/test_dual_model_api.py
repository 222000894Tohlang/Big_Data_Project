
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"
IMAGE_PATH = Path(__file__).resolve().parent / "test_images" / "tiger.jpg"


def test_model_registry():
    response = requests.get(
        f"{BASE_URL}/models",
        timeout=15,
    )

    assert response.status_code == 200, response.text

    models = response.json()["models"]

    assert "cnn" in models
    assert "svm" in models

    print("PASS: Model registry")
    print(f"CNN: {models['cnn']['status']}")
    print(f"SVM: {models['svm']['status']}")


def test_cnn_prediction():
    with IMAGE_PATH.open("rb") as image:
        response = requests.post(
            f"{BASE_URL}/predict",
            files={"file": ("tiger.jpg", image, "image/jpeg")},
            timeout=60,
        )

    assert response.status_code == 200, response.text

    result = response.json()

    assert "species" in result
    assert "confidence" in result

    print("\nPASS: CNN prediction")
    print(f"Species: {result['species']}")
    print(f"Confidence: {result['confidence']}")


def test_svm_unavailable():
    with IMAGE_PATH.open("rb") as image:
        response = requests.post(
            f"{BASE_URL}/predict/svm",
            files={"file": ("tiger.jpg", image, "image/jpeg")},
            timeout=30,
        )

    assert response.status_code == 503, response.text

    print("\nPASS: SVM unavailable response")
    print(f"Status: {response.status_code}")
    print(f"Message: {response.json()['detail']}")


def test_health():
    response = requests.get(
        f"{BASE_URL}/health",
        timeout=15,
    )

    assert response.status_code == 200, response.text

    print("\nPASS: API health")


if __name__ == "__main__":
    print("\nDual-Model API Integration Tests\n")

    test_health()
    test_model_registry()
    test_cnn_prediction()
    test_svm_unavailable()

    print("\nAll dual-model API tests passed.")

