
from pathlib import Path

from inference import predict_species


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "test_images"

MODELS = ["cnn", "cnn_leakage_aware"]

TEST_IMAGES = {
    "tiger.jpg": "tiger",
    "leopard.jpg": "leopard",
    "panda.jpg": "panda",
}


def main():
    print("\nCNN Model Comparison\n")

    results = {}

    for model_name in MODELS:
        correct = 0

        print(f"\nModel: {model_name}")
        print("-" * 55)

        for filename, expected in TEST_IMAGES.items():
            image_bytes = (
                IMAGE_DIR / filename
            ).read_bytes()

            prediction = predict_species(
                image_bytes,
                model_name=model_name,
            )

            predicted = prediction["species"]
            confidence = prediction["confidence"]

            is_correct = predicted.lower() == expected

            if is_correct:
                correct += 1

            print(
                f"{filename:15} "
                f"Expected: {expected:10} "
                f"Predicted: {predicted:12} "
                f"Confidence: {confidence:.2%} "
                f"{'PASS' if is_correct else 'FAIL'}"
            )

        results[model_name] = correct

    print("\nSummary")
    print("-" * 35)

    for model_name, correct in results.items():
        print(
            f"{model_name}: "
            f"{correct}/{len(TEST_IMAGES)} correct"
        )


if __name__ == "__main__":
    main()
