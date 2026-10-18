
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"

TEST_IMAGES = {
    "tiger.jpg": "tiger",
    "leopard.jpg": "leopard",
    "panda.jpg": "panda",
}

IMAGE_DIRECTORY = (
    Path(__file__).resolve().parent / "test_images"
)


def test_cnn_predictions():
    print("\nCNN Prediction Verification\n")

    correct = 0

    for filename, expected_species in TEST_IMAGES.items():
        image_path = IMAGE_DIRECTORY / filename

        with image_path.open("rb") as image:
            response = requests.post(
                f"{BASE_URL}/predict",
                files={
                    "file": (
                        filename,
                        image,
                        "image/jpeg",
                    )
                },
                timeout=60,
            )

        response.raise_for_status()

        result = response.json()

        predicted_species = str(
            result["species"]
        ).strip().lower()

        confidence = float(result["confidence"])

        is_correct = predicted_species == expected_species

        if is_correct:
            correct += 1

        print(f"Image: {filename}")
        print(f"Expected: {expected_species}")
        print(f"Predicted: {predicted_species}")
        print(f"Confidence: {confidence:.4f}")
        print(
            f"Result: {'PASS' if is_correct else 'FAIL'}"
        )
        print("-" * 40)

    total = len(TEST_IMAGES)

    print(f"\nCorrect predictions: {correct}/{total}")
    print(f"Small-sample accuracy: {correct / total:.2%}")


if __name__ == "__main__":
    test_cnn_predictions()
