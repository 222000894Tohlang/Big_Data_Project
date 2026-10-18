from pathlib import Path

import requests


API_URL = "http://127.0.0.1:8000"

TEST_IMAGES_DIR = Path(__file__).parent / "test_images"


def test_health():

    response = requests.get(
        f"{API_URL}/health",
        timeout=10,
    )

    assert response.status_code == 200

    print("PASS: API health check")


def test_model_health():

    response = requests.get(
        f"{API_URL}/health/model",
        timeout=30,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"

    print("PASS: CNN model health check")


def test_prediction(image_path: Path):

    with open(image_path, "rb") as image_file:

        response = requests.post(
            f"{API_URL}/predict",
            files={
                "file": (
                    image_path.name,
                    image_file,
                    "image/jpeg",
                )
            },
            timeout=60,
        )

    assert response.status_code == 200, response.text

    data = response.json()

    assert "timing" in data

    timing = data["timing"]

    assert timing["preprocessing_ms"] >= 0
    assert timing["inference_ms"] >= 0
    assert timing["total_ms"] >= 0

    print("  Preprocessing:", timing["preprocessing_ms"], "ms")
    print("  CNN inference:", timing["inference_ms"], "ms")
    print("  Total processing:", timing["total_ms"], "ms")

    assert "species" in data
    assert "confidence" in data
    assert "top_predictions" in data

    assert len(data["top_predictions"]) == 3

    assert data["species"] == (data["top_predictions"][0]["species"])

    assert 0 <= data["confidence"] <= 1

    probabilities = [item["confidence"] for item in data["top_predictions"]]

    assert probabilities == sorted(
        probabilities,
        reverse=True,
    )

    print(f"PASS: {image_path.name}")
    print(f"  Species: {data['species']}")
    print(f"  Confidence: {data['confidence']:.2%}")


def test_invalid_file():

    response = requests.post(
        f"{API_URL}/predict",
        files={
            "file": (
                "invalid.txt",
                b"This is not an image.",
                "text/plain",
            )
        },
        timeout=10,
    )

    assert response.status_code == 415

    print("PASS: Invalid file rejected")


def test_corrupted_image():

    response = requests.post(
        f"{API_URL}/predict",
        files={
            "file": (
                "corrupted.jpg",
                b"This is not a real JPEG.",
                "image/jpeg",
            )
        },
        timeout=10,
    )

    assert response.status_code == 400

    print("PASS: Corrupted image rejected")


def test_oversized_image():

    response = requests.post(
        f"{API_URL}/predict",
        files={
            "file": (
                "large.jpg",
                b"x" * (10 * 1024 * 1024 + 1),
                "image/jpeg",
            )
        },
        timeout=30,
    )

    assert response.status_code == 413

    print("PASS: Oversized image rejected")


if __name__ == "__main__":

    print("\nRunning Wildlife API Integration Tests\n")

    test_health()
    test_model_health()

    for image_path in sorted(TEST_IMAGES_DIR.glob("*.jpg")):
        test_prediction(image_path)

    test_invalid_file()
    test_corrupted_image()
    test_oversized_image()

    print("\nAll executed integration tests passed!")
